#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/kitti_player_with_gt.py
#
# Finish-first (realtime) mode:
# - publish PointCloud2 + frame_idx + done only
# - NO GT generation, NO bbox markers
# - use simulated stamp derived from frame_idx (recommended)

from __future__ import annotations

import os
import glob
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time

from std_msgs.msg import Header, Int32, Bool
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from kitti_roi_eval.offline_generate_gt_bbox import load_tracklets_xml, boxes_at_frame


def qos_sensor(depth: int = 5) -> QoSProfile:
    # SensorData相当（BestEffort）: 大容量点群で詰まらせない
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def qos_reliable(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


class KittiPlayerWithGT(Node):
    """
    Realtime/finish-first KITTI player (raw velodyne .bin):
      - publish PointCloud2 on points_topic
      - publish frame_idx (Int32) on frame_idx_topic
      - publish done (Bool) once at end
    """

    def __init__(self):
        super().__init__("kitti_player_with_gt")

        # ---- params ----
        self.declare_parameter("drive_dir", "")
        self.declare_parameter("points_topic", "/livox/lidar_perturbed")
        self.declare_parameter("frame_idx_topic", "kitti_player/frame_idx")
        self.declare_parameter("done_topic", "kitti_player/done")

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("start_idx", 0)
        self.declare_parameter("end_idx", -1)   # -1 => last
        self.declare_parameter("stride", 1)

        self.declare_parameter("use_sim_stamp", True)
        self.declare_parameter("end_grace_sec", 2.0)

        self.declare_parameter("qos_depth_pc", 5)
        self.declare_parameter("qos_depth_misc", 10)
        # Optional bbox line marker publishing (default OFF to avoid side effects)
        self.declare_parameter("publish_bbox_markers", False)
        self.declare_parameter("bbox_topic", "kitti_player/bbox_lines")
        self.declare_parameter("bbox_tracklet_xml", "")
        self.declare_parameter("bbox_tracklet_z_is_bottom", True)
        self.declare_parameter("bbox_classes", ["Car", "Van", "Truck", "Pedestrian", "Cyclist"])
        self.declare_parameter("bbox_line_width", 0.08)
        self.declare_parameter("bbox_color_r", 1.0)
        self.declare_parameter("bbox_color_g", 0.2)
        self.declare_parameter("bbox_color_b", 0.2)
        self.declare_parameter("bbox_color_a", 1.0)

        # ---- read params ----
        drive_dir = str(self.get_parameter("drive_dir").value).strip()
        if drive_dir == "":
            raise RuntimeError("drive_dir is empty.")
        self.drive_dir = os.path.expanduser(drive_dir)

        self.points_topic = str(self.get_parameter("points_topic").value)
        self.frame_idx_topic = str(self.get_parameter("frame_idx_topic").value)
        self.done_topic = str(self.get_parameter("done_topic").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.start_idx = int(self.get_parameter("start_idx").value)
        self.end_idx = int(self.get_parameter("end_idx").value)
        self.stride = int(self.get_parameter("stride").value)

        self.use_sim_stamp = bool(self.get_parameter("use_sim_stamp").value)
        self.end_grace_sec = float(self.get_parameter("end_grace_sec").value)

        qos_depth_pc = int(self.get_parameter("qos_depth_pc").value)
        qos_depth_misc = int(self.get_parameter("qos_depth_misc").value)
        self.publish_bbox_markers = bool(self.get_parameter("publish_bbox_markers").value)
        self.bbox_topic = str(self.get_parameter("bbox_topic").value)
        self.bbox_tracklet_xml = str(self.get_parameter("bbox_tracklet_xml").value).strip()
        self.bbox_tracklet_z_is_bottom = bool(self.get_parameter("bbox_tracklet_z_is_bottom").value)
        self.bbox_classes = [str(x) for x in self.get_parameter("bbox_classes").value]
        self.bbox_line_width = float(self.get_parameter("bbox_line_width").value)
        self.bbox_color = ColorRGBA(
            r=float(self.get_parameter("bbox_color_r").value),
            g=float(self.get_parameter("bbox_color_g").value),
            b=float(self.get_parameter("bbox_color_b").value),
            a=float(self.get_parameter("bbox_color_a").value),
        )

        # ---- locate .bin files ----
        velo_dir = os.path.join(self.drive_dir, "velodyne_points", "data")
        self.point_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
        if len(self.point_files) == 0:
            raise FileNotFoundError(f"no .bin found: {velo_dir}")

        last = len(self.point_files) - 1
        if self.end_idx < 0:
            self.end_idx = last
        self.end_idx = min(self.end_idx, last)
        self.start_idx = max(0, self.start_idx)

        if self.start_idx > self.end_idx:
            raise RuntimeError(f"invalid range: start_idx({self.start_idx}) > end_idx({self.end_idx})")

        # ---- pubs ----
        self.pub_pc = self.create_publisher(PointCloud2, self.points_topic, qos_sensor(qos_depth_pc))
        self.pub_idx = self.create_publisher(Int32, self.frame_idx_topic, qos_reliable(qos_depth_misc))
        self.pub_done = self.create_publisher(Bool, self.done_topic, qos_reliable(qos_depth_misc))
        self.pub_bbox = None
        self.tracklets = []
        if self.publish_bbox_markers:
            tracklet_xml = self.bbox_tracklet_xml
            if tracklet_xml == "":
                tracklet_xml = os.path.join(self.drive_dir, "tracklet_labels.xml")
            tracklet_xml = os.path.expanduser(tracklet_xml)
            try:
                self.tracklets = load_tracklets_xml(tracklet_xml)
                self.pub_bbox = self.create_publisher(Marker, self.bbox_topic, qos_reliable(qos_depth_misc))
            except Exception as e:
                self.get_logger().warn(
                    f"publish_bbox_markers=true but failed to load tracklets ({tracklet_xml}): {e}. "
                    "Disable bbox marker publishing."
                )
                self.publish_bbox_markers = False

        # ---- state ----
        self.cur = int(self.start_idx)
        self._done_sent = False
        self._end_timer = None

        period = 1.0 / max(1e-6, self.rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            "KittiPlayerWithGT (finish-first realtime) started.\n"
            f"  drive_dir={self.drive_dir}\n"
            f"  files={len(self.point_files)} (idx: 0..{last})\n"
            f"  range: start_idx={self.start_idx} end_idx={self.end_idx} stride={self.stride}\n"
            f"  rate_hz={self.rate_hz}\n"
            f"  use_sim_stamp={self.use_sim_stamp}\n"
            f"  topics: points={self.points_topic}, frame_idx={self.frame_idx_topic}, done={self.done_topic}\n"
            f"  publish_bbox_markers={self.publish_bbox_markers} topic={self.bbox_topic}\n"
        )

    def _box_corners_3d(self, box) -> np.ndarray:
        l2 = float(box.l) * 0.5
        w2 = float(box.w) * 0.5
        h2 = float(box.h) * 0.5
        corners_local = np.array(
            [
                [l2, w2, h2],
                [l2, -w2, h2],
                [-l2, -w2, h2],
                [-l2, w2, h2],
                [l2, w2, -h2],
                [l2, -w2, -h2],
                [-l2, -w2, -h2],
                [-l2, w2, -h2],
            ],
            dtype=np.float64,
        )
        c = math.cos(float(box.yaw))
        s = math.sin(float(box.yaw))
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        t = np.array([float(box.cx), float(box.cy), float(box.cz)], dtype=np.float64)
        return (corners_local @ r.T) + t[None, :]

    def _publish_bbox_marker(self, header: Header, frame_idx: int):
        if not self.publish_bbox_markers or self.pub_bbox is None:
            return
        boxes = boxes_at_frame(
            self.tracklets,
            int(frame_idx),
            tracklet_z_is_bottom=self.bbox_tracklet_z_is_bottom,
            classes=self.bbox_classes,
        )
        marker = Marker()
        marker.header = header
        marker.ns = "kitti_bbox"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = float(max(0.001, self.bbox_line_width))
        marker.color = self.bbox_color
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        # 12 box edges (pairs of corner indices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        pts = []
        for b in boxes:
            c = self._box_corners_3d(b)
            for i, j in edges:
                p1 = Point(x=float(c[i, 0]), y=float(c[i, 1]), z=float(c[i, 2]))
                p2 = Point(x=float(c[j, 0]), y=float(c[j, 1]), z=float(c[j, 2]))
                pts.extend([p1, p2])
        marker.points = pts
        self.pub_bbox.publish(marker)

    def _make_sim_stamp(self, frame_idx: int) -> Time:
        # t = frame_idx / rate_hz
        dt = 1.0 / max(1e-9, self.rate_hz)
        t = float(frame_idx) * dt
        sec = int(t)
        nsec = int((t - sec) * 1e9)
        return Time(seconds=sec, nanoseconds=nsec)

    def _publish_done_and_shutdown(self):
        if self._done_sent:
            return
        self._done_sent = True

        msg = Bool()
        msg.data = True
        self.pub_done.publish(msg)

        try:
            self.timer.cancel()
        except Exception:
            pass

        def _shutdown_once():
            if self._end_timer is not None:
                try:
                    self._end_timer.cancel()
                except Exception:
                    pass
            try:
                self.destroy_node()
            except Exception:
                pass
            if rclpy.ok():
                rclpy.shutdown()

        self._end_timer = self.create_timer(max(0.1, self.end_grace_sec), _shutdown_once)

    def _on_timer(self):
        if self.cur > self.end_idx:
            self.get_logger().info("Reached end. Publish done and stop.")
            self._publish_done_and_shutdown()
            return

        # header stamp
        if self.use_sim_stamp:
            stamp = self._make_sim_stamp(self.cur).to_msg()
        else:
            stamp = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = stamp
        header.frame_id = "velo"

        # frame_idx
        idx = Int32()
        idx.data = int(self.cur)
        self.pub_idx.publish(idx)

        # load & publish points
        path = self.point_files[self.cur]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
        xyz = pts[:, :3].astype(np.float32, copy=False)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc2 = point_cloud2.create_cloud(header, fields, xyz)
        self.pub_pc.publish(pc2)
        self._publish_bbox_marker(header, self.cur)

        self.cur += max(1, int(self.stride))


def main():
    rclpy.init()
    node = KittiPlayerWithGT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
