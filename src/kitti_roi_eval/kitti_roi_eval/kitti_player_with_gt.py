#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/kitti_player_with_gt.py

from __future__ import annotations

import os
import math
import glob
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py import point_cloud2

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


# --------------------------
# QoS helper
# --------------------------
def make_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


# --------------------------
# Tracklet structures
# --------------------------
@dataclass
class TrackletPose:
    # KITTI raw tracklet pose (tx,ty,tz, rx,ry,rz)
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float


@dataclass
class Tracklet:
    obj_type: str
    h: float
    w: float
    l: float
    first_frame: int
    poses: List[TrackletPose]


@dataclass
class Box3D:
    obj_type: str
    # center (velo)
    cx: float
    cy: float
    cz: float
    # size (l,w,h)
    l: float
    w: float
    h: float
    # yaw around z
    yaw: float


def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def _rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def _wrap_deg_0_360(x: float) -> float:
    y = x % 360.0
    if y < 0:
        y += 360.0
    return y


# --------------------------
# Tracklet parser (KITTI raw)
# --------------------------
def load_tracklets_xml(xml_path: str) -> List[Tracklet]:
    """
    KITTI raw: tracklet_labels.xml
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"tracklet xml not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracklets: List[Tracklet] = []

    # The structure is typically:
    # <tracklets>
    #   <count>...</count>
    #   <item_version>...</item_version>
    #   <item> ... </item>
    #   <item> ... </item>
    # </tracklets>
    items = root.findall("item")
    if len(items) == 0:
        # sometimes root is <tracklets> and items are under <tracklets>/<item>
        items = root.findall(".//item")

    for it in items:
        obj_type = (it.findtext("objectType") or "unknown").strip()
        h = float(it.findtext("h") or 0.0)
        w = float(it.findtext("w") or 0.0)
        l = float(it.findtext("l") or 0.0)
        first_frame = int(it.findtext("first_frame") or 0)

        poses: List[TrackletPose] = []
        poses_node = it.find("poses")
        if poses_node is None:
            continue

        pose_items = poses_node.findall("item")
        for p in pose_items:
            tx = float(p.findtext("tx") or 0.0)
            ty = float(p.findtext("ty") or 0.0)
            tz = float(p.findtext("tz") or 0.0)
            rx = float(p.findtext("rx") or 0.0)
            ry = float(p.findtext("ry") or 0.0)
            rz = float(p.findtext("rz") or 0.0)
            poses.append(TrackletPose(tx, ty, tz, rx, ry, rz))

        tracklets.append(Tracklet(obj_type=obj_type, h=h, w=w, l=l, first_frame=first_frame, poses=poses))

    return tracklets


def boxes_at_frame(tracklets: List[Tracklet], frame_idx: int, tracklet_z_is_bottom: bool = True) -> List[Box3D]:
    """
    frame_idx に存在する tracklet を Box3D に変換して返す（座標系は velo を想定）
    tracklet_z_is_bottom=True の場合、tz を「底面中心」とみなし、cz = tz + h/2 に補正。
    """
    out: List[Box3D] = []
    for tr in tracklets:
        k = frame_idx - tr.first_frame
        if k < 0 or k >= len(tr.poses):
            continue
        pose = tr.poses[k]
        cx, cy, cz = pose.tx, pose.ty, pose.tz
        if tracklet_z_is_bottom:
            cz = cz + tr.h * 0.5

        # KITTI raw tracklet の rz を yaw(z) として扱う（必要なら後で調整）
        yaw = pose.rz

        out.append(Box3D(
            obj_type=tr.obj_type,
            cx=cx, cy=cy, cz=cz,
            l=tr.l, w=tr.w, h=tr.h,
            yaw=yaw
        ))
    return out


# --------------------------
# Box utils (velo)
# --------------------------
def rotz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def box_corners_3d(box: Box3D) -> np.ndarray:
    """
    returns (8,3) corners in world coords.
    order: 0..3 top face, 4..7 bottom face (consistent edges list)
    """
    l2 = box.l * 0.5
    w2 = box.w * 0.5
    h2 = box.h * 0.5

    # local corners (x forward, y left, z up)
    # top z=+h2, bottom z=-h2
    corners_local = np.array([
        [ l2,  w2,  h2],
        [ l2, -w2,  h2],
        [-l2, -w2,  h2],
        [-l2,  w2,  h2],
        [ l2,  w2, -h2],
        [ l2, -w2, -h2],
        [-l2, -w2, -h2],
        [-l2,  w2, -h2],
    ], dtype=np.float64)

    R = rotz(box.yaw)
    t = np.array([box.cx, box.cy, box.cz], dtype=np.float64)
    corners = (corners_local @ R.T) + t[None, :]
    return corners


def point_in_box(p: np.ndarray, box: Box3D) -> bool:
    """
    p: (3,) world
    """
    R = rotz(box.yaw)
    t = np.array([box.cx, box.cy, box.cz], dtype=np.float64)
    q = p - t
    q_local = (R.T @ q.reshape(3, 1)).reshape(3,)
    return (abs(q_local[0]) <= box.l * 0.5 and
            abs(q_local[1]) <= box.w * 0.5 and
            abs(q_local[2]) <= box.h * 0.5)


def angle_bin_indices(x: float, y: float, z: float, H: int, V: int,
                      hfov_deg: float, vfov_deg: float) -> Optional[Tuple[int, int]]:
    """
    KITTI raw の velo を想定:
    azimuth: atan2(y, x) -> [0,360)
    elevation: atan2(z, sqrt(x^2+y^2)) -> [-vfov/2, +vfov/2] を想定
    """
    r_xy = math.hypot(x, y)
    if r_xy < 1e-9:
        return None
    az = _wrap_deg_0_360(_rad2deg(math.atan2(y, x)))
    el = _rad2deg(math.atan2(z, r_xy))

    # horizontal: [0, hfov)
    if hfov_deg <= 0:
        return None
    h = int((az / hfov_deg) * H)
    if h < 0 or h >= H:
        return None

    # vertical: [-vfov/2, +vfov/2]
    vmin = -vfov_deg * 0.5
    vmax = +vfov_deg * 0.5
    if el < vmin or el > vmax:
        return None
    v = int(((el - vmin) / vfov_deg) * V)
    if v < 0 or v >= V:
        return None

    return (v, h)


# --------------------------
# Marker utils
# --------------------------
def make_bbox_line_marker(header: Header, ns: str, mid: int, corners: np.ndarray,
                          rgba: Tuple[float, float, float, float],
                          line_width: float) -> Marker:
    """
    corners: (8,3)
    """
    m = Marker()
    m.header = header
    m.ns = ns
    m.id = int(mid)
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = float(line_width)
    m.color.r = float(rgba[0])
    m.color.g = float(rgba[1])
    m.color.b = float(rgba[2])
    m.color.a = float(rgba[3])
    m.lifetime.sec = 0
    m.lifetime.nanosec = 0

    # 12 edges: (top 4) + (bottom 4) + (vertical 4)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # top
        (4, 5), (5, 6), (6, 7), (7, 4),   # bottom
        (0, 4), (1, 5), (2, 6), (3, 7),   # vertical
    ]
    pts: List[Point] = []
    for a, b in edges:
        pa = corners[a]
        pb = corners[b]
        pts.append(Point(x=float(pa[0]), y=float(pa[1]), z=float(pa[2])))
        pts.append(Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2])))
    m.points = pts
    return m


def make_text_marker(header: Header, ns: str, mid: int, x: float, y: float, z: float,
                     text: str, rgba: Tuple[float, float, float, float],
                     scale: float = 0.8) -> Marker:
    m = Marker()
    m.header = header
    m.ns = ns
    m.id = int(mid)
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(x)
    m.pose.position.y = float(y)
    m.pose.position.z = float(z)
    m.scale.z = float(scale)
    m.color.r = float(rgba[0])
    m.color.g = float(rgba[1])
    m.color.b = float(rgba[2])
    m.color.a = float(rgba[3])
    m.text = str(text)
    return m


# --------------------------
# Node
# --------------------------
class KittiPlayerWithGT(Node):
    def __init__(self):
        super().__init__("kitti_player_with_gt")

        # -------- parameters --------
        self.declare_parameter("drive_dir", "")
        self.declare_parameter("points_topic", "/livox/lidar_perturbed")
        self.declare_parameter("gt_topic", "pc_perturber/gt_mask_mono8")

        # bins / fov
        self.declare_parameter("num_horizontal_bins", 128)
        self.declare_parameter("num_vertical_bins", 128)
        self.declare_parameter("horizontal_fov_deg", 360.0)
        self.declare_parameter("vertical_fov_deg", 60.0)

        # frame range
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)     # -1 => last
        self.declare_parameter("stride", 1)

        # publish rate
        self.declare_parameter("publish_hz", 10.0)

        # GT mask
        self.declare_parameter("gt_enable", True)
        self.declare_parameter("gt_value", 255)

        # bbox publish (RViz)
        self.declare_parameter("bbox_enable", True)
        self.declare_parameter("bbox_topic", "kitti_player/bbox_markers")
        self.declare_parameter("bbox_ns", "kitti_bbox")
        self.declare_parameter("bbox_line_width", 0.05)
        self.declare_parameter("bbox_rgba", [1.0, 1.0, 0.0, 0.9])  # yellow-ish
        self.declare_parameter("bbox_text_enable", True)
        self.declare_parameter("bbox_text_scale", 0.8)

        # tracklet option
        self.declare_parameter("tracklet_z_is_bottom", True)
        self.declare_parameter("frame_id", "velo")  # Fixed Frame = velo

        # -------- read parameters --------
        self.drive_dir = str(self.get_parameter("drive_dir").value).strip()
        if self.drive_dir == "":
            raise RuntimeError("drive_dir is empty. Please set drive_dir to KITTI raw drive path.")

        self.points_topic = str(self.get_parameter("points_topic").value)
        self.gt_topic = str(self.get_parameter("gt_topic").value)

        self.H = int(self.get_parameter("num_horizontal_bins").value)
        self.V = int(self.get_parameter("num_vertical_bins").value)
        self.hfov_deg = float(self.get_parameter("horizontal_fov_deg").value)
        self.vfov_deg = float(self.get_parameter("vertical_fov_deg").value)

        self.start_frame = int(self.get_parameter("start_frame").value)
        self.end_frame = int(self.get_parameter("end_frame").value)
        self.stride = int(self.get_parameter("stride").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.gt_enable = bool(self.get_parameter("gt_enable").value)
        self.gt_value = int(self.get_parameter("gt_value").value)

        self.bbox_enable = bool(self.get_parameter("bbox_enable").value)
        self.bbox_topic = str(self.get_parameter("bbox_topic").value)
        self.bbox_ns = str(self.get_parameter("bbox_ns").value)
        self.bbox_line_width = float(self.get_parameter("bbox_line_width").value)
        rgba_list = self.get_parameter("bbox_rgba").value
        if isinstance(rgba_list, (list, tuple)) and len(rgba_list) == 4:
            self.bbox_rgba = (float(rgba_list[0]), float(rgba_list[1]), float(rgba_list[2]), float(rgba_list[3]))
        else:
            self.bbox_rgba = (1.0, 1.0, 0.0, 0.9)
        self.bbox_text_enable = bool(self.get_parameter("bbox_text_enable").value)
        self.bbox_text_scale = float(self.get_parameter("bbox_text_scale").value)

        self.tracklet_z_is_bottom = bool(self.get_parameter("tracklet_z_is_bottom").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        # -------- locate data --------
        velo_dir = os.path.join(self.drive_dir, "velodyne_points", "data")
        self.point_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
        if len(self.point_files) == 0:
            raise FileNotFoundError(f"no velodyne .bin found: {velo_dir}")

        if self.end_frame < 0:
            self.end_frame = len(self.point_files) - 1
        self.end_frame = min(self.end_frame, len(self.point_files) - 1)

        # tracklets
        tracklet_xml = os.path.join(self.drive_dir, "tracklet_labels.xml")
        self.tracklets = load_tracklets_xml(tracklet_xml)

        # -------- publishers/sub --------
        qos = make_qos(10)
        self.pub_pc = self.create_publisher(PointCloud2, self.points_topic, qos)
        self.pub_gt = self.create_publisher(Image, self.gt_topic, qos) if self.gt_enable else None
        self.pub_bbox = self.create_publisher(MarkerArray, self.bbox_topic, qos) if self.bbox_enable else None

        # -------- state --------
        self.cur_frame = self.start_frame
        self.timer = self.create_timer(1.0 / max(0.1, self.publish_hz), self._on_timer)

        self.get_logger().info(
            "KittiPlayerWithGT started.\n"
            f"  drive_dir={self.drive_dir}\n"
            f"  points_topic={self.points_topic}\n"
            f"  gt_topic={self.gt_topic}\n"
            f"  bbox_topic={self.bbox_topic}\n"
            f"  frames={len(self.point_files)} start={self.start_frame} end={self.end_frame} stride={self.stride}\n"
            f"  bins=VxH={self.V}x{self.H} fov(H,V)=({self.hfov_deg},{self.vfov_deg})\n"
            f"  frame_id={self.frame_id}\n"
            f"  tracklet_z_is_bottom={self.tracklet_z_is_bottom}\n"
        )

    # --------------------------
    # Timer callback
    # --------------------------
    def _on_timer(self):
        if self.cur_frame > self.end_frame:
            self.get_logger().info("Reached end. Stop timer.")
            self.timer.cancel()
            return

        stamp = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id

        # ---- load points ----
        path = self.point_files[self.cur_frame]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
        xyz = pts[:, :3]

        # ---- publish pointcloud ----
        pc2 = self._make_pc2(header, xyz)
        self.pub_pc.publish(pc2)

        # ---- boxes at frame ----
        boxes = boxes_at_frame(self.tracklets, self.cur_frame, tracklet_z_is_bottom=self.tracklet_z_is_bottom)

        # ---- publish bbox markers ----
        if self.pub_bbox is not None:
            ma = MarkerArray()
            mid = 0
            for b in boxes:
                corners = box_corners_3d(b)
                ma.markers.append(make_bbox_line_marker(header, self.bbox_ns, mid, corners, self.bbox_rgba, self.bbox_line_width))
                mid += 1
                if self.bbox_text_enable:
                    ma.markers.append(make_text_marker(
                        header, self.bbox_ns + "_text", mid,
                        b.cx, b.cy, b.cz + b.h * 0.6,
                        b.obj_type, (1.0, 1.0, 1.0, 0.9),
                        scale=self.bbox_text_scale
                    ))
                    mid += 1
            self.pub_bbox.publish(ma)

        # ---- publish GT mask (angle-bin) ----
        if self.pub_gt is not None:
            gt = self._make_gt_mask_from_points(xyz, boxes)
            msg = Image()
            msg.header = header
            msg.height = int(self.V)
            msg.width = int(self.H)
            msg.encoding = "mono8"
            msg.is_bigendian = 0
            msg.step = int(self.H)
            msg.data = gt.tobytes()
            self.pub_gt.publish(msg)

        # advance
        self.cur_frame += max(1, self.stride)

    # --------------------------
    # Builders
    # --------------------------
    def _make_pc2(self, header: Header, xyz: np.ndarray) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        points_iter = xyz.astype(np.float32)
        return point_cloud2.create_cloud(header, fields, points_iter)

    def _make_gt_mask_from_points(self, xyz: np.ndarray, boxes: List[Box3D]) -> np.ndarray:
        """
        点群から「bbox内部に入った点の角度ビン」を GT として立てる版。
        （GTの正当性確認用にシンプルな実装。）
        """
        gt = np.zeros((self.V, self.H), dtype=np.uint8)
        if len(boxes) == 0:
            return gt

        for p in xyz:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            vh = angle_bin_indices(x, y, z, self.H, self.V, self.hfov_deg, self.vfov_deg)
            if vh is None:
                continue
            v, h = vh

            pw = np.array([x, y, z], dtype=np.float64)
            inside = False
            for b in boxes:
                if point_in_box(pw, b):
                    inside = True
                    break
            if inside:
                gt[v, h] = np.uint8(self.gt_value)

        return gt


def main():
    rclpy.init()
    node = KittiPlayerWithGT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
