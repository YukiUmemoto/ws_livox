#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/gen_gt_binmask.py

from __future__ import annotations

import os
import math
import glob
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# --------------------------
# Tracklet structures
# --------------------------
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


# --------------------------
# Helpers
# --------------------------
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

    tracklets: List[Tracklet] = []
    items = root.findall("item")
    if len(items) == 0:
        items = root.findall(".//item")

    for it in items:
        obj_type = (it.findtext("objectType") or "unknown").strip()
        h = float(it.findtext("h") or 0.0)
        w = float(it.findtext("w") or 0.0)
        l = float(it.findtext("l") or 0.0)
        first_frame = int(it.findtext("first_frame") or 0)

        poses_node = it.find("poses")
        if poses_node is None:
            continue

        poses: List[TrackletPose] = []
        for p in poses_node.findall("item"):
            poses.append(
                TrackletPose(
                    tx=float(p.findtext("tx") or 0.0),
                    ty=float(p.findtext("ty") or 0.0),
                    tz=float(p.findtext("tz") or 0.0),
                    rx=float(p.findtext("rx") or 0.0),
                    ry=float(p.findtext("ry") or 0.0),
                    rz=float(p.findtext("rz") or 0.0),
                )
            )

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

        # 既存実装に合わせて yaw=rz を使用（必要ならここを差し替え）
        yaw = pose.rz

        out.append(Box3D(tr.obj_type, cx, cy, cz, tr.l, tr.w, tr.h, yaw))
    return out


def rotz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def box_corners_3d(box: Box3D) -> np.ndarray:
    l2 = box.l * 0.5
    w2 = box.w * 0.5
    h2 = box.h * 0.5

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
    return (corners_local @ R.T) + t[None, :]


def corners_to_az_el_deg(corners: np.ndarray) -> Tuple[List[float], List[float]]:
    azs: List[float] = []
    els: List[float] = []
    for p in corners:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        r_xy = math.hypot(x, y)
        if r_xy < 1e-9:
            continue
        az = _wrap_deg_0_360(_rad2deg(math.atan2(y, x)))
        el = _rad2deg(math.atan2(z, r_xy))
        azs.append(az)
        els.append(el)
    return azs, els


def _az_interval_minspan(azs: List[float]) -> Tuple[float, float]:
    """
    azs in [0,360). wrap を考慮して，角度スパンが最小になる区間 [a0,a1] を返す（a1は a0以上，場合により >360）
    """
    a = np.array(azs, dtype=np.float64)
    if a.size == 0:
        return (0.0, -1.0)

    a2 = a.copy()
    a2[a2 < 180.0] += 360.0

    span1 = float(a.max() - a.min())
    span2 = float(a2.max() - a2.min())
    if span1 <= span2:
        return (float(a.min()), float(a.max()))
    else:
        return (float(a2.min()), float(a2.max()))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _dilate_bool(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return m
    out = m.astype(bool)
    for _ in range(r):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        nxt = out.copy()
        for dv in (-1, 0, 1):
            for dh in (-1, 0, 1):
                if dv == 0 and dh == 0:
                    continue
                sl_v = slice(1 + dv, 1 + dv + out.shape[0])
                sl_h = slice(1 + dh, 1 + dh + out.shape[1])
                nxt |= padded[sl_v, sl_h]
        out = nxt
    return out


def _align_binmap(m: np.ndarray, flip_ud: bool, flip_lr: bool, center_az: bool) -> np.ndarray:
    out = m
    if flip_ud:
        out = out[::-1, :]
    if flip_lr:
        out = out[:, ::-1]
    if center_az:
        out = np.roll(out, shift=(out.shape[1] // 2), axis=1)
    return out


def gen_gt_mask_angular(
    boxes: List[Box3D],
    V: int,
    H: int,
    hfov_deg: float,
    vfov_deg: float,
    vfov_up_deg: float | None = None,
    vfov_down_deg: float | None = None,
) -> np.ndarray:
    gt = np.zeros((V, H), dtype=np.uint8)
    if len(boxes) == 0:
        return gt

    if vfov_up_deg is not None and vfov_down_deg is not None and (vfov_up_deg + vfov_down_deg) > 0:
        vmin = -float(vfov_down_deg)
        vmax = +float(vfov_up_deg)
        vfov_deg = float(vfov_up_deg) + float(vfov_down_deg)
    else:
        vmin = -vfov_deg * 0.5
        vmax = +vfov_deg * 0.5

    for b in boxes:
        corners = box_corners_3d(b)
        azs, els = corners_to_az_el_deg(corners)
        if len(azs) == 0 or len(els) == 0:
            continue

        a0, a1 = _az_interval_minspan(azs)
        if a1 < a0:
            continue

        e0 = float(min(els))
        e1 = float(max(els))

        # FOV clip
        e0 = _clip(e0, vmin, vmax)
        e1 = _clip(e1, vmin, vmax)
        if e1 < e0:
            continue

        # v range
        vv0 = int(((e0 - vmin) / vfov_deg) * V)
        vv1 = int(((e1 - vmin) / vfov_deg) * V)
        vv0 = max(0, min(V - 1, vv0))
        vv1 = max(0, min(V - 1, vv1))

        # h range (wrap handled by allowing a0,a1>360)
        hh0 = int((a0 / hfov_deg) * H)
        hh1 = int((a1 / hfov_deg) * H)

        # inclusive fill
        for hh in range(hh0, hh1 + 1):
            h = hh % H
            gt[vv0:vv1 + 1, h] = 255

    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_dir", required=True)
    ap.add_argument("--tracklet_xml", default="")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--hfov_deg", type=float, default=360.0)
    ap.add_argument("--vfov_deg", type=float, default=26.8)
    ap.add_argument("--vfov_up_deg", type=float, default=2.0)
    ap.add_argument("--vfov_down_deg", type=float, default=24.8)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--classes", nargs="*", default=["Car", "Van", "Truck", "Pedestrian", "Cyclist"])
    ap.add_argument("--tracklet_z_is_bottom", action="store_true", default=True)
    ap.add_argument("--dilate_r", type=int, default=0)
    ap.add_argument("--align_flip_ud", action="store_true", default=False)
    ap.add_argument("--align_flip_lr", action="store_true", default=False)
    ap.add_argument("--align_center_az", action="store_true", default=False)
    args = ap.parse_args()

    drive_dir = os.path.expanduser(args.drive_dir)
    velo_dir = os.path.join(drive_dir, "velodyne_points", "data")
    point_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
    if len(point_files) == 0:
        raise FileNotFoundError(f"no velodyne .bin found: {velo_dir}")

    tracklet_xml = args.tracklet_xml.strip()
    if tracklet_xml == "":
        tracklet_xml = os.path.join(drive_dir, "tracklet_labels.xml")
    tracklet_xml = os.path.expanduser(tracklet_xml)

    tracklets = load_tracklets_xml(tracklet_xml)

    start = int(args.start_idx)
    end = int(args.end_idx)
    if end < 0:
        end = len(point_files) - 1
    end = min(end, len(point_files) - 1)

    frames = list(range(start, end + 1, max(1, int(args.stride))))
    gt_stack = np.zeros((len(frames), int(args.V), int(args.H)), dtype=np.uint8)

    for i, fidx in enumerate(frames):
        boxes = boxes_at_frame(
            tracklets, fidx,
            tracklet_z_is_bottom=bool(args.tracklet_z_is_bottom),
            classes=[str(x) for x in args.classes],
        )
        vfov_deg = float(args.vfov_deg)
        vfov_up = float(args.vfov_up_deg)
        vfov_down = float(args.vfov_down_deg)
        if vfov_up + vfov_down <= 0:
            vfov_up = vfov_deg * 0.5
            vfov_down = vfov_deg * 0.5
        gt = gen_gt_mask_angular(
            boxes,
            args.V,
            args.H,
            args.hfov_deg,
            vfov_up + vfov_down,
            vfov_up_deg=vfov_up,
            vfov_down_deg=vfov_down,
        )
        if args.dilate_r > 0:
            gt = (_dilate_bool(gt > 0, args.dilate_r).astype(np.uint8) * 255)
        gt = _align_binmap(
            gt,
            flip_ud=bool(args.align_flip_ud),
            flip_lr=bool(args.align_flip_lr),
            center_az=bool(args.align_center_az),
        )
        gt_stack[i] = gt

        if (i % 20) == 0:
            print(f"[gen_gt] {i}/{len(frames)} frame={fidx} boxes={len(boxes)} gt_bins={int(np.sum(gt>0))}")

    out_npz = os.path.expanduser(args.out_npz)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        gt=gt_stack,
        frame_indices=np.array(frames, dtype=np.int32),
        V=np.int32(args.V),
        H=np.int32(args.H),
        hfov_deg=np.float64(args.hfov_deg),
        vfov_deg=np.float64(args.vfov_deg),
        dilate_r=np.int32(args.dilate_r),
        align_flip_ud=np.uint8(1 if args.align_flip_ud else 0),
        align_flip_lr=np.uint8(1 if args.align_flip_lr else 0),
        align_center_az=np.uint8(1 if args.align_center_az else 0),
        classes=np.array([str(x) for x in args.classes], dtype=object),
        drive_dir=np.array(drive_dir, dtype=object),
        tracklet_xml=np.array(tracklet_xml, dtype=object),
    )
    print(f"[gen_gt] wrote: {out_npz}  (frames={len(frames)}, VxH={args.V}x{args.H})")


if __name__ == "__main__":
    main()
