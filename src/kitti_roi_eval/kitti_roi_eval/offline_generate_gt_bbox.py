#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/offline_generate_gt_bbox.py
#
# Offline heavy processing:
# - generate GT masks (mono8 VxH) from KITTI raw velodyne + tracklets
# - export bbox per frame to JSON
#
# Example:
#   python3 -m kitti_roi_eval.offline_generate_gt_bbox \
#     --drive_dir /path/to/2011_09_26_drive_0011_sync \
#     --out_dir /path/to/out_gt \
#     --start 0 --end 232 --stride 1 --V 128 --H 128 --vfov 60 --hfov 360 \
#     --classes Car Van Truck Pedestrian Cyclist

from __future__ import annotations

import os
import glob
import json
import math
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np


@dataclass
class TrackletPose:
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
    cx: float
    cy: float
    cz: float
    l: float
    w: float
    h: float
    yaw: float


def _rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def _wrap_deg_0_360(x: float) -> float:
    y = x % 360.0
    if y < 0:
        y += 360.0
    return y


def load_tracklets_xml(xml_path: str) -> List[Tracklet]:
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"tracklet xml not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    items = root.findall("item")
    if len(items) == 0:
        items = root.findall(".//item")

    tracklets: List[Tracklet] = []
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
        for p in poses_node.findall("item"):
            poses.append(TrackletPose(
                tx=float(p.findtext("tx") or 0.0),
                ty=float(p.findtext("ty") or 0.0),
                tz=float(p.findtext("tz") or 0.0),
                rx=float(p.findtext("rx") or 0.0),
                ry=float(p.findtext("ry") or 0.0),
                rz=float(p.findtext("rz") or 0.0),
            ))
        tracklets.append(Tracklet(obj_type=obj_type, h=h, w=w, l=l, first_frame=first_frame, poses=poses))

    return tracklets


def boxes_at_frame(tracklets: List[Tracklet], frame_idx: int,
                   tracklet_z_is_bottom: bool = True,
                   classes: Optional[List[str]] = None) -> List[Box3D]:
    out: List[Box3D] = []
    for tr in tracklets:
        if classes and tr.obj_type not in classes:
            continue
        k = frame_idx - tr.first_frame
        if k < 0 or k >= len(tr.poses):
            continue
        pose = tr.poses[k]
        cx, cy, cz = pose.tx, pose.ty, pose.tz
        if tracklet_z_is_bottom:
            cz = cz + tr.h * 0.5
        yaw = pose.rz
        out.append(Box3D(tr.obj_type, cx, cy, cz, tr.l, tr.w, tr.h, yaw))
    return out


def rotz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def point_in_box_batch(points_xyz: np.ndarray, box: Box3D) -> np.ndarray:
    """
    points_xyz: (N,3)
    return: (N,) bool inside
    """
    R = rotz(box.yaw)
    t = np.array([box.cx, box.cy, box.cz], dtype=np.float64)
    q = points_xyz.astype(np.float64) - t[None, :]
    q_local = (q @ R)  # since q_local = R.T @ q ; use q @ R is equivalent when R orthonormal
    # NOTE: Here R is rotation matrix; (q @ R) == (R.T @ q.T).T
    inside = (
        (np.abs(q_local[:, 0]) <= box.l * 0.5) &
        (np.abs(q_local[:, 1]) <= box.w * 0.5) &
        (np.abs(q_local[:, 2]) <= box.h * 0.5)
    )
    return inside


def angle_bins(
    xyz: np.ndarray,
    H: int,
    V: int,
    hfov_deg: float,
    vfov_deg: float,
    vfov_up_deg: float | None = None,
    vfov_down_deg: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    xyz: (N,3)
    returns:
      valid: (N,) bool
      v_idx: (N,) int
      h_idx: (N,) int
    """
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    rxy = np.hypot(x, y)
    valid = rxy > 1e-9

    az = np.zeros_like(rxy)
    el = np.zeros_like(rxy)

    az[valid] = np.arctan2(y[valid], x[valid])
    el[valid] = np.arctan2(z[valid], rxy[valid])

    az_deg = _wrap_deg_0_360(_rad2deg(az))
    el_deg = _rad2deg(el)

    # horizontal [0, hfov)
    h = np.floor((az_deg / hfov_deg) * H).astype(np.int64)
    valid &= (h >= 0) & (h < H)

    # vertical
    if vfov_up_deg is not None and vfov_down_deg is not None and (vfov_up_deg + vfov_down_deg) > 0:
        vmin = -float(vfov_down_deg)
        vmax = +float(vfov_up_deg)
        vfov_deg = float(vfov_up_deg) + float(vfov_down_deg)
    else:
        vmin = -vfov_deg * 0.5
        vmax = +vfov_deg * 0.5
    v = np.floor(((el_deg - vmin) / vfov_deg) * V).astype(np.int64)
    valid &= (el_deg >= vmin) & (el_deg <= vmax) & (v >= 0) & (v < V)

    return valid, v, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_dir", required=True)
    ap.add_argument("--tracklet_xml", default="")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--vfov", type=float, default=26.8)
    ap.add_argument("--vfov_up", type=float, default=2.0)
    ap.add_argument("--vfov_down", type=float, default=24.8)
    ap.add_argument("--hfov", type=float, default=360.0)

    ap.add_argument("--gt_value", type=int, default=255)
    ap.add_argument("--tracklet_z_is_bottom", action="store_true")

    ap.add_argument("--classes", nargs="*", default=["Car", "Van", "Truck", "Pedestrian", "Cyclist"])

    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--save_png_dir", action="store_true")
    ap.add_argument("--save_bbox_json", action="store_true")

    args = ap.parse_args()

    drive_dir = os.path.expanduser(args.drive_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    velo_dir = os.path.join(drive_dir, "velodyne_points", "data")
    bin_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
    if len(bin_files) == 0:
        raise FileNotFoundError(f"no .bin found: {velo_dir}")

    last = len(bin_files) - 1
    start = max(0, args.start)
    end = last if args.end < 0 else min(args.end, last)
    stride = max(1, args.stride)

    tracklet_xml = args.tracklet_xml.strip()
    if tracklet_xml == "":
        tracklet_xml = os.path.join(drive_dir, "tracklet_labels.xml")
    tracklets = load_tracklets_xml(tracklet_xml)

    classes = [str(c) for c in (args.classes or [])]

    frames = list(range(start, end + 1, stride))
    V, H = int(args.V), int(args.H)

    gt_stack = []
    bbox_out: Dict[int, List[Dict]] = {}

    png_dir = os.path.join(out_dir, "gt_png")
    if args.save_png_dir:
        os.makedirs(png_dir, exist_ok=True)

    for fi in frames:
        pts = np.fromfile(bin_files[fi], dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]

        boxes = boxes_at_frame(tracklets, fi, tracklet_z_is_bottom=args.tracklet_z_is_bottom, classes=classes)

        gt = np.zeros((V, H), dtype=np.uint8)
        if len(boxes) > 0 and xyz.shape[0] > 0:
            valid, v_idx, h_idx = angle_bins(
                xyz,
                H=H,
                V=V,
                hfov_deg=args.hfov,
                vfov_deg=args.vfov,
                vfov_up_deg=args.vfov_up,
                vfov_down_deg=args.vfov_down,
            )
            xyz_valid = xyz[valid]
            v_valid = v_idx[valid]
            h_valid = h_idx[valid]

            inside_any = np.zeros((xyz_valid.shape[0],), dtype=bool)
            # heavy but offline OK
            for b in boxes:
                inside_any |= point_in_box_batch(xyz_valid, b)

            vv = v_valid[inside_any]
            hh = h_valid[inside_any]
            gt[vv, hh] = np.uint8(args.gt_value)

        gt_stack.append(gt)

        if args.save_bbox_json:
            bbox_out[fi] = [
                dict(
                    obj_type=b.obj_type,
                    cx=float(b.cx), cy=float(b.cy), cz=float(b.cz),
                    l=float(b.l), w=float(b.w), h=float(b.h),
                    yaw=float(b.yaw),
                )
                for b in boxes
            ]

        if args.save_png_dir:
            # write as raw mono8 png via imageio if available; fallback numpy save
            try:
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(png_dir, f"gt_{fi:06d}.png"), gt)
            except Exception:
                np.save(os.path.join(png_dir, f"gt_{fi:06d}.npy"), gt)

        if (fi - start) % max(1, 10 * stride) == 0:
            print(f"[offline_gt] frame {fi}/{end} boxes={len(boxes)}")

    gt_arr = np.stack(gt_stack, axis=0)  # (T,V,H)

    if args.save_npz:
        npz_path = os.path.join(out_dir, f"gt_masks_V{V}_H{H}_f{start}-{end}_s{stride}.npz")
        np.savez_compressed(npz_path, gt=gt_arr, start=start, end=end, stride=stride, V=V, H=H,
                            hfov=args.hfov, vfov=args.vfov, classes=classes)
        print(f"[offline_gt] wrote npz: {npz_path}")

    if args.save_bbox_json:
        json_path = os.path.join(out_dir, f"bbox_per_frame_f{start}-{end}_s{stride}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bbox_out, f, ensure_ascii=False, indent=2)
        print(f"[offline_gt] wrote bbox json: {json_path}")

    print("[offline_gt] done.")


if __name__ == "__main__":
    main()
