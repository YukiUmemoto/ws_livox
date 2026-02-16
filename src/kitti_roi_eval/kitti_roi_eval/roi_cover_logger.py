#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/roi_cover_logger.py

from __future__ import annotations

import os
import csv
import time
from collections import deque
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Int32
from sensor_msgs.msg import Image, PointCloud2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def make_qos(depth: int = 50, reliable: bool = False) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def img_to_bool_mono8(msg: Image, expected_shape: Tuple[int, int]) -> np.ndarray:
    if msg.encoding != "mono8":
        raise RuntimeError(f"Expected mono8, got {msg.encoding}")
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
    if a.shape != expected_shape:
        raise RuntimeError(f"Shape mismatch: got {a.shape}, expected {expected_shape}")
    return (a > 0)


def stamp_key_from_header(h) -> Tuple[int, int]:
    return (int(h.stamp.sec), int(h.stamp.nanosec))


def stamp_key_img(msg: Image) -> Tuple[int, int]:
    return stamp_key_from_header(msg.header)


def stamp_key_pc(msg: PointCloud2) -> Tuple[int, int]:
    return stamp_key_from_header(msg.header)


def pc2_num_points(msg: PointCloud2) -> int:
    ps = int(msg.point_step)
    if ps <= 0:
        return int(msg.width) * int(msg.height)
    return int(len(msg.data) // ps)


def _dilate_bool(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return m.astype(bool)
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


def _align_bool_mask(m: np.ndarray, flip_ud: bool, flip_lr: bool, center_az: bool) -> np.ndarray:
    out = m
    if flip_ud:
        out = out[::-1, :]
    if flip_lr:
        out = out[:, ::-1]
    if center_az:
        out = np.roll(out, shift=(out.shape[1] // 2), axis=1)
    return out


class RoiCoverLogger(Node):
    """
    frame_idx を基準に，
      - pred (mono8)
      - omega (mono8)
      - points_count (Int32)
      - 事前生成GT (npz)
    から BBOXカバー率（= tp/gt_bins on Omega）を算出し，CSVへ確実に記録する。
    finish-first 用に ack_frame_idx を返す。
    """

    def __init__(self):
        super().__init__("roi_cover_logger")

        # -------------------------
        # parameters
        # -------------------------
        self.declare_parameter("pred_topic", "roi_est/roi_imp_mono8")
        self.declare_parameter("pred_topic_aux", "")
        self.declare_parameter("pred_combine_mode", "single")  # single | and | or | xor
        self.declare_parameter("omega_topic", "roi_est/omega_mono8")
        self.declare_parameter("gt_mask_topic", "")  # optional dynamic GT mask topic
        self.declare_parameter("gt_combine_mode", "single")  # single | and | or | xor
        self.declare_parameter("pc_topic", "/livox/lidar_perturbed")
        self.declare_parameter("frame_idx_topic", "kitti_player/frame_idx")
        self.declare_parameter("points_count_topic", "kitti_player/points_count")
        self.declare_parameter("ack_topic", "kitti_player/ack_frame_idx")

        self.declare_parameter("gt_npz", "")
        self.declare_parameter("cover_tol_bins", 0)
        self.declare_parameter("align_use_gt_meta", True)
        self.declare_parameter("align_flip_ud", False)
        self.declare_parameter("align_flip_lr", False)
        self.declare_parameter("align_center_az", False)

        self.declare_parameter("out_dir", "")
        self.declare_parameter("csv_name", "cover_per_frame.csv")
        self.declare_parameter("sync_policy", "arrival")  # arrival | strict_stamp
        self.declare_parameter("sequential_mode", False)
        self.declare_parameter("write_missing_tail_rows", True)
        self.declare_parameter("viz_enable", True)
        self.declare_parameter("viz_max_frames", 100000)
        self.declare_parameter("cache_max_entries", 300)

        self.declare_parameter("qos_sub_depth", 50)
        self.declare_parameter("qos_sub_reliable", False)     # pred/omega は BEST_EFFORT に合わせるのが基本
        self.declare_parameter("qos_meta_depth", 50)
        self.declare_parameter("qos_meta_reliable", False)    # frame_idx / points_count も BEST_EFFORT に合わせる

        self.declare_parameter("qos_ack_depth", 50)
        self.declare_parameter("qos_ack_reliable", True)

        # -------------------------
        # read params
        # -------------------------
        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.pred_topic_aux = str(self.get_parameter("pred_topic_aux").value).strip()
        self.pred_combine_mode = str(self.get_parameter("pred_combine_mode").value).strip().lower()
        self.omega_topic = str(self.get_parameter("omega_topic").value)
        self.gt_mask_topic = str(self.get_parameter("gt_mask_topic").value).strip()
        self.gt_combine_mode = str(self.get_parameter("gt_combine_mode").value).strip().lower()
        self.pc_topic = str(self.get_parameter("pc_topic").value)
        self.frame_idx_topic = str(self.get_parameter("frame_idx_topic").value)
        self.points_count_topic = str(self.get_parameter("points_count_topic").value)
        self.ack_topic = str(self.get_parameter("ack_topic").value)

        self.cover_tol_bins = int(self.get_parameter("cover_tol_bins").value)

        out_dir = str(self.get_parameter("out_dir").value).strip()
        if out_dir == "":
            out_dir = os.path.abspath("result_kitti_cover")
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, str(self.get_parameter("csv_name").value))
        self.sync_policy = str(self.get_parameter("sync_policy").value).strip().lower()
        if self.sync_policy not in ("arrival", "strict_stamp"):
            self.get_logger().warn(f"Unknown sync_policy='{self.sync_policy}', fallback to 'arrival'.")
            self.sync_policy = "arrival"
        self.sequential_mode = bool(self.get_parameter("sequential_mode").value)
        self.write_missing_tail_rows = bool(self.get_parameter("write_missing_tail_rows").value)
        if self.sync_policy == "strict_stamp":
            if self.sequential_mode:
                self.get_logger().warn(
                    "sync_policy=strict_stamp ignores sequential_mode. Force sequential_mode=False."
                )
            self.sequential_mode = False
        self.viz_enable = bool(self.get_parameter("viz_enable").value)
        self.viz_max_frames = int(self.get_parameter("viz_max_frames").value)
        self.cache_max_entries = int(self.get_parameter("cache_max_entries").value)
        self.viz_saved = 0
        self.viz_dir = os.path.join(self.out_dir, "viz")
        if self.viz_enable:
            os.makedirs(self.viz_dir, exist_ok=True)

        gt_npz = str(self.get_parameter("gt_npz").value).strip()
        if gt_npz == "":
            raise RuntimeError("gt_npz is empty. Please set gt_npz to precomputed GT npz path.")
        self.gt_npz = os.path.expanduser(gt_npz)

        # -------------------------
        # load GT
        # -------------------------
        z = np.load(self.gt_npz, allow_pickle=True)
        self.gt_stack = z["gt"].astype(np.uint8)                 # (N,V,H)
        self.gt_frames = z["frame_indices"].astype(np.int32)     # (N,)
        self.V = int(z["V"])
        self.H = int(z["H"])
        self.expected_shape = (self.V, self.H)
        self.gt_map: Dict[int, int] = {int(f): i for i, f in enumerate(self.gt_frames)}

        use_meta = bool(self.get_parameter("align_use_gt_meta").value)
        meta_flip_ud = bool(int(z["align_flip_ud"])) if (use_meta and "align_flip_ud" in z) else False
        meta_flip_lr = bool(int(z["align_flip_lr"])) if (use_meta and "align_flip_lr" in z) else False
        meta_center_az = bool(int(z["align_center_az"])) if (use_meta and "align_center_az" in z) else False
        self.align_flip_ud = bool(self.get_parameter("align_flip_ud").value) or meta_flip_ud
        self.align_flip_lr = bool(self.get_parameter("align_flip_lr").value) or meta_flip_lr
        self.align_center_az = bool(self.get_parameter("align_center_az").value) or meta_center_az

        # -------------------------
        # QoS
        # -------------------------
        sub_depth = int(self.get_parameter("qos_sub_depth").value)
        sub_rel = bool(self.get_parameter("qos_sub_reliable").value)
        qos_roi = make_qos(depth=sub_depth, reliable=sub_rel)

        meta_depth = int(self.get_parameter("qos_meta_depth").value)
        meta_rel = bool(self.get_parameter("qos_meta_reliable").value)
        qos_meta = make_qos(depth=meta_depth, reliable=meta_rel)

        ack_depth = int(self.get_parameter("qos_ack_depth").value)
        ack_rel = bool(self.get_parameter("qos_ack_reliable").value)
        qos_ack = make_qos(depth=ack_depth, reliable=ack_rel)

        # -------------------------
        # pubs/subs (single set)
        # -------------------------
        self.sub_frame_idx = self.create_subscription(Int32, self.frame_idx_topic, self._cb_frame_idx, qos_meta)
        self.sub_points_count = self.create_subscription(Int32, self.points_count_topic, self._cb_points_count, qos_meta)
        self.sub_pc = self.create_subscription(PointCloud2, self.pc_topic, self._cb_pc, qos_roi)
        self.sub_pred = self.create_subscription(Image, self.pred_topic, self._cb_pred, qos_roi)
        self.sub_pred_aux = None
        if self.pred_topic_aux != "":
            self.sub_pred_aux = self.create_subscription(Image, self.pred_topic_aux, self._cb_pred_aux, qos_roi)
        self.sub_omega = self.create_subscription(Image, self.omega_topic, self._cb_omega, qos_roi)
        self.sub_gt_mask = None
        if self.gt_mask_topic != "":
            self.sub_gt_mask = self.create_subscription(Image, self.gt_mask_topic, self._cb_gt_mask, qos_roi)
        self.pub_ack = self.create_publisher(Int32, self.ack_topic, qos_ack)

        # -------------------------
        # state
        # -------------------------
        self.cur_frame: Optional[int] = None
        self.t_frame: Optional[float] = None
        self.points_count: Optional[int] = None

        self.pred: Optional[np.ndarray] = None
        self.pred_aux: Optional[np.ndarray] = None
        self.gt_dyn: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
        self.t_pred: Optional[float] = None
        self.t_pred_aux: Optional[float] = None
        self.t_gt_dyn: Optional[float] = None
        self.t_omega: Optional[float] = None

        self.done = set()
        self.frame_queue = deque()
        self.points_queue = deque()
        self.pred_queue = deque()
        self.pred_aux_queue = deque()
        self.gt_dyn_queue = deque()
        self.omega_queue = deque()
        self.cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.pc_stamp_queue: deque[Tuple[int, int]] = deque()
        self.frame_idx_to_stamp: Dict[int, Tuple[int, int]] = {}
        self.stamp_to_frame_idx: Dict[Tuple[int, int], int] = {}
        self.stamp_points: Dict[Tuple[int, int], int] = {}
        self.evaluated_frames = set()

        # -------------------------
        # CSV open
        # -------------------------
        self._fp = open(self.csv_path, "w", newline="")
        self._wr = csv.writer(self._fp)
        self._wr.writerow([
            "frame_idx",
            "points_count",
            "omega_bins",
            "pred_bins_on_omega",
            "pred_ratio_on_omega",
            "gt_bins_on_omega",
            "gt_ratio_on_omega",
            "tp",
            "gt_cover_ratio",
            "gt_cover_ratio_tol",
            "precision",
            "recall",
            "f1",
            "iou",
            "lat_pred_ms_from_frame",
            "lat_omega_ms_from_frame",
            "eval_ms",
            "gt_missing",
            "gt_empty",
        ])
        self._fp.flush()

        self.get_logger().info(
            "RoiCoverLogger started.\n"
            f"  pred={self.pred_topic}\n"
            f"  pred_aux={self.pred_topic_aux if self.pred_topic_aux != '' else '(none)'}\n"
            f"  pred_combine_mode={self.pred_combine_mode}\n"
            f"  omega={self.omega_topic}\n"
            f"  gt_mask_topic={self.gt_mask_topic if self.gt_mask_topic != '' else '(none)'}\n"
            f"  gt_combine_mode={self.gt_combine_mode}\n"
            f"  pc={self.pc_topic}\n"
            f"  frame_idx={self.frame_idx_topic}\n"
            f"  points_count={self.points_count_topic}\n"
            f"  ack={self.ack_topic}\n"
            f"  gt_npz={self.gt_npz} (frames={len(self.gt_frames)}, VxH={self.V}x{self.H})\n"
            f"  align: flip_ud={self.align_flip_ud} flip_lr={self.align_flip_lr} center_az={self.align_center_az}\n"
            f"  cover_tol_bins={self.cover_tol_bins}\n"
            f"  out_dir={self.out_dir}\n"
            f"  csv={self.csv_path}\n"
            f"  sync_policy={self.sync_policy}\n"
            f"  viz_enable={self.viz_enable} (max_frames={self.viz_max_frames})\n"
            f"  sequential_mode={self.sequential_mode}\n"
            f"  cache_max_entries={self.cache_max_entries}\n"
            f"  qos_roi  reliable={sub_rel} depth={sub_depth}\n"
            f"  qos_meta reliable={meta_rel} depth={meta_depth}\n"
            f"  qos_ack  reliable={ack_rel} depth={ack_depth}\n"
        )

    def _ack(self, frame_idx: int):
        m = Int32()
        m.data = int(frame_idx)
        self.pub_ack.publish(m)

    def _reset_for_frame(self, frame_idx: int):
        self.cur_frame = int(frame_idx)
        self.t_frame = time.time()
        self.points_count = None
        self.pred = None
        self.pred_aux = None
        self.gt_dyn = None
        self.omega = None
        self.t_pred = None
        self.t_pred_aux = None
        self.t_gt_dyn = None
        self.t_omega = None

    def _cb_frame_idx(self, msg: Int32):
        f = int(msg.data)
        if f in self.done:
            # ACKロスト対策
            self._ack(f)
            return

        if self.sync_policy == "strict_stamp":
            self.frame_queue.append(f)
            paired = self._pair_frameidx_with_pc_stamp()
            for key in paired:
                self._try_eval_strict_for_key(key)
            return

        if self.sequential_mode:
            self.frame_queue.append(f)
            self._try_eval_sequential()
            return

        if self.cur_frame is None or f != self.cur_frame:
            self._reset_for_frame(f)

    def _cb_points_count(self, msg: Int32):
        if self.sync_policy == "strict_stamp":
            # strict_stamp では PointCloud2 から points_count を採る
            return
        if self.sequential_mode:
            self.points_queue.append(int(msg.data))
            self._try_eval_sequential()
            return
        if self.cur_frame is None:
            return
        self.points_count = int(msg.data)

    def _cb_pc(self, msg: PointCloud2):
        key = stamp_key_pc(msg)
        n_points = int(pc2_num_points(msg))
        self.stamp_points[key] = n_points
        d = self.cache.get(key)
        if d is None:
            d = {}
            self.cache[key] = d
        d["points_count"] = n_points

        if self.sync_policy == "strict_stamp":
            self.pc_stamp_queue.append(key)
            paired = self._pair_frameidx_with_pc_stamp()
            for pkey in paired:
                self._try_eval_strict_for_key(pkey)

        self._prune_cache()

    def _cb_pred(self, msg: Image):
        if self.sync_policy == "strict_stamp":
            self._store_mask_strict("pred", msg)
            return
        if self.sequential_mode:
            try:
                pred = img_to_bool_mono8(msg, self.expected_shape)
                if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                    pred = _align_bool_mask(pred, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
                self.pred_queue.append((pred, time.time()))
            except Exception as e:
                self.get_logger().error(f"failed to parse pred: {e}")
                return
            self._try_eval_sequential()
            return

        if self.cur_frame is None:
            return
        try:
            self.pred = img_to_bool_mono8(msg, self.expected_shape)
            if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                self.pred = _align_bool_mask(self.pred, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
            self.t_pred = time.time()
        except Exception as e:
            self.get_logger().error(f"failed to parse pred: {e}")
            return
        self._try_eval()

    def _cb_pred_aux(self, msg: Image):
        if self.sync_policy == "strict_stamp":
            self._store_mask_strict("pred_aux", msg)
            return
        if self.sequential_mode:
            try:
                pred_aux = img_to_bool_mono8(msg, self.expected_shape)
                if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                    pred_aux = _align_bool_mask(pred_aux, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
                self.pred_aux_queue.append((pred_aux, time.time()))
            except Exception as e:
                self.get_logger().error(f"failed to parse pred_aux: {e}")
                return
            self._try_eval_sequential()
            return

        if self.cur_frame is None:
            return
        try:
            self.pred_aux = img_to_bool_mono8(msg, self.expected_shape)
            if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                self.pred_aux = _align_bool_mask(self.pred_aux, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
            self.t_pred_aux = time.time()
        except Exception as e:
            self.get_logger().error(f"failed to parse pred_aux: {e}")
            return
        self._try_eval()

    def _cb_omega(self, msg: Image):
        if self.sync_policy == "strict_stamp":
            self._store_mask_strict("omega", msg)
            return
        if self.sequential_mode:
            try:
                omega = img_to_bool_mono8(msg, self.expected_shape)
                if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                    omega = _align_bool_mask(omega, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
                self.omega_queue.append((omega, time.time()))
            except Exception as e:
                self.get_logger().error(f"failed to parse omega: {e}")
                return
            self._try_eval_sequential()
            return

        if self.cur_frame is None:
            return
        try:
            self.omega = img_to_bool_mono8(msg, self.expected_shape)
            if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                self.omega = _align_bool_mask(self.omega, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
            self.t_omega = time.time()
        except Exception as e:
            self.get_logger().error(f"failed to parse omega: {e}")
            return
        self._try_eval()

    def _cb_gt_mask(self, msg: Image):
        if self.sync_policy == "strict_stamp":
            self._store_mask_strict("gt_dyn", msg)
            return
        if self.sequential_mode:
            try:
                gt_dyn = img_to_bool_mono8(msg, self.expected_shape)
                if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                    gt_dyn = _align_bool_mask(gt_dyn, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
                self.gt_dyn_queue.append((gt_dyn, time.time()))
            except Exception as e:
                self.get_logger().error(f"failed to parse gt_mask: {e}")
                return
            self._try_eval_sequential()
            return

        if self.cur_frame is None:
            return
        try:
            self.gt_dyn = img_to_bool_mono8(msg, self.expected_shape)
            if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                self.gt_dyn = _align_bool_mask(self.gt_dyn, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
            self.t_gt_dyn = time.time()
        except Exception as e:
            self.get_logger().error(f"failed to parse gt_mask: {e}")
            return
        self._try_eval()

    def _try_eval(self):
        if self.cur_frame is None or self.t_frame is None:
            return
        f = int(self.cur_frame)
        if f in self.done:
            return
        if self.pred is None or self.omega is None:
            return
        if self.pred_topic_aux != "" and self.pred_aux is None:
            return
        if self.gt_mask_topic != "" and self.gt_dyn is None:
            return

        self._eval_one(
            frame_idx=f,
            pred=self.pred,
            pred_aux=self.pred_aux,
            gt_dyn=self.gt_dyn,
            omega=self.omega,
            points_count=(int(self.points_count) if self.points_count is not None else -1),
            t_frame=self.t_frame,
            t_pred=self.t_pred,
            t_pred_aux=self.t_pred_aux,
            t_gt_dyn=self.t_gt_dyn,
            t_omega=self.t_omega,
        )

    def _try_eval_sequential(self):
        while len(self.frame_queue) > 0 and len(self.pred_queue) > 0 and len(self.omega_queue) > 0:
            if self.pred_topic_aux != "" and len(self.pred_aux_queue) == 0:
                break
            if self.gt_mask_topic != "" and len(self.gt_dyn_queue) == 0:
                break
            f = int(self.frame_queue.popleft())
            if f in self.done:
                self._ack(f)
                continue
            pred, t_pred = self.pred_queue.popleft()
            pred_aux = None
            t_pred_aux = None
            if self.pred_topic_aux != "":
                pred_aux, t_pred_aux = self.pred_aux_queue.popleft()
            gt_dyn = None
            t_gt_dyn = None
            if self.gt_mask_topic != "":
                gt_dyn, t_gt_dyn = self.gt_dyn_queue.popleft()
            omega, t_omega = self.omega_queue.popleft()
            points_count = int(self.points_queue.popleft()) if len(self.points_queue) > 0 else -1
            # sequential mode では frame_idx 受信時刻を基準時刻とみなす
            t_frame = time.time()
            self._eval_one(
                frame_idx=f,
                pred=pred,
                pred_aux=pred_aux,
                gt_dyn=gt_dyn,
                omega=omega,
                points_count=points_count,
                t_frame=t_frame,
                t_pred=t_pred,
                t_pred_aux=t_pred_aux,
                t_gt_dyn=t_gt_dyn,
                t_omega=t_omega,
            )

    def _prune_cache(self):
        while len(self.cache) > max(10, self.cache_max_entries):
            oldest = next(iter(self.cache))
            self.cache.pop(oldest, None)

    def _pair_frameidx_with_pc_stamp(self) -> List[Tuple[int, int]]:
        paired: List[Tuple[int, int]] = []
        while len(self.frame_queue) > 0 and len(self.pc_stamp_queue) > 0:
            f = int(self.frame_queue.popleft())
            if f in self.done:
                continue
            skey = self.pc_stamp_queue.popleft()
            self.frame_idx_to_stamp[f] = skey
            self.stamp_to_frame_idx[skey] = f
            paired.append(skey)
        return paired

    def _store_mask_strict(self, kind: str, msg: Image):
        key = stamp_key_img(msg)
        try:
            mask = img_to_bool_mono8(msg, self.expected_shape)
            if self.align_flip_ud or self.align_flip_lr or self.align_center_az:
                mask = _align_bool_mask(mask, self.align_flip_ud, self.align_flip_lr, self.align_center_az)
        except Exception as e:
            self.get_logger().error(f"failed to parse {kind}: {e}")
            return

        d = self.cache.get(key)
        if d is None:
            d = {}
            self.cache[key] = d
        d[kind] = mask

        self._prune_cache()
        self._try_eval_strict_for_key(key)

    def _try_eval_strict_for_key(self, key: Tuple[int, int]):
        if key not in self.stamp_to_frame_idx:
            return
        f = int(self.stamp_to_frame_idx[key])
        if f in self.done:
            self._ack(f)
            return
        d = self.cache.get(key)
        if d is None:
            return
        if ("pred" not in d) or ("omega" not in d):
            return
        if self.pred_topic_aux != "" and ("pred_aux" not in d):
            return
        if self.gt_mask_topic != "" and ("gt_dyn" not in d):
            return
        self._eval_one(
            frame_idx=f,
            pred=d["pred"],
            pred_aux=d.get("pred_aux"),
            gt_dyn=d.get("gt_dyn"),
            omega=d["omega"],
            points_count=int(d.get("points_count", self.stamp_points.get(key, -1))),
            t_frame=None,
            t_pred=None,
            t_pred_aux=None,
            t_gt_dyn=None,
            t_omega=None,
        )
        self.evaluated_frames.add(f)
        self.cache.pop(key, None)

    def _eval_one(
        self,
        frame_idx: int,
        pred: np.ndarray,
        pred_aux: Optional[np.ndarray],
        gt_dyn: Optional[np.ndarray],
        omega: np.ndarray,
        points_count: int,
        t_frame: Optional[float],
        t_pred: Optional[float],
        t_pred_aux: Optional[float],
        t_gt_dyn: Optional[float],
        t_omega: Optional[float],
    ):
        f = int(frame_idx)
        if f in self.done:
            self._ack(f)
            return

        t0 = time.time()

        valid = omega
        p = self._combine_mask((pred & valid), (pred_aux & valid) if pred_aux is not None else None, self.pred_combine_mode, "pred")

        gi = self.gt_map.get(f, None)
        gt_missing = 0
        if gi is None:
            gt_missing = 1
            g_bbox = np.zeros_like(valid, dtype=bool)
        else:
            g_bbox = (self.gt_stack[gi] > 0) & valid
        g = self._combine_mask(g_bbox, (gt_dyn & valid) if gt_dyn is not None else None, self.gt_combine_mode, "gt")

        gt_bins = int(np.sum(g))
        gt_empty = 1 if gt_bins == 0 else 0

        tp = int(np.sum(p & g))
        fp = int(np.sum(p & (~g)))
        fn = int(np.sum((~p) & g))

        eps = 1e-9
        union = tp + fp + fn
        iou = 1.0 if union == 0 else tp / (union + eps)

        precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp + eps)
        recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn + eps)
        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall + eps)

        omega_bins = int(np.sum(valid))
        pred_bins = int(np.sum(p))
        pred_ratio = 0.0 if omega_bins == 0 else pred_bins / (omega_bins + eps)
        gt_ratio = 0.0 if omega_bins == 0 else gt_bins / (omega_bins + eps)

        # GTに対するカバー率（= tp/gt_bins）
        gt_cover_ratio = 1.0 if gt_bins == 0 else tp / (gt_bins + eps)

        # 近傍許容（pred膨張）
        if self.cover_tol_bins > 0:
            p_tol = _dilate_bool(p, self.cover_tol_bins)
            tp_tol = int(np.sum(p_tol & g))
            gt_cover_ratio_tol = 1.0 if gt_bins == 0 else tp_tol / (gt_bins + eps)
        else:
            gt_cover_ratio_tol = gt_cover_ratio

        lat_omega_ms = -1.0 if t_omega is None or t_frame is None else (t_omega - t_frame) * 1000.0
        pred_ts = t_pred
        if t_pred_aux is not None:
            pred_ts = max(pred_ts if pred_ts is not None else t_pred_aux, t_pred_aux)
        lat_pred_ms = -1.0 if pred_ts is None or t_frame is None else (pred_ts - t_frame) * 1000.0
        eval_ms = (time.time() - t0) * 1000.0

        row = [
            f,
            int(points_count),
            omega_bins,
            pred_bins,
            float(pred_ratio),
            gt_bins,
            float(gt_ratio),
            tp,
            float(gt_cover_ratio),
            float(gt_cover_ratio_tol),
            float(precision),
            float(recall),
            float(f1),
            float(iou),
            float(lat_pred_ms),
            float(lat_omega_ms),
            float(eval_ms),
            int(gt_missing),
            int(gt_empty),
        ]
        self._wr.writerow(row)
        self._fp.flush()

        if self.viz_enable and self.viz_saved < self.viz_max_frames:
            self._save_viz(int(f), p, g, valid)
            self.viz_saved += 1

        self.done.add(f)
        self._ack(f)

        if (len(self.done) % 20) == 0:
            self.get_logger().info(
                f"[cover] done={len(self.done)} frame={f} "
                f"cover={gt_cover_ratio:.3f} (tol={gt_cover_ratio_tol:.3f}) "
                f"pred_ratio={pred_ratio:.3f} gt_ratio={gt_ratio:.3f}"
            )

    def _combine_mask(
        self,
        a: np.ndarray,
        b: Optional[np.ndarray],
        mode: str,
        label: str,
    ) -> np.ndarray:
        m = mode.lower().strip()
        if m == "single":
            return a
        if b is None:
            return a
        if m == "and":
            return a & b
        if m == "or":
            return a | b
        if m == "xor":
            return np.logical_xor(a, b)
        self.get_logger().warn(f"Unknown {label}_combine_mode='{mode}', fallback to 'single'.")
        return a

    def _save_viz(self, frame_idx: int, p: np.ndarray, g: np.ndarray, valid: np.ndarray):
        tp = p & g
        fp = p & (~g)
        fn = (~p) & g

        vis = np.zeros_like(p, dtype=np.uint8)
        vis[tp] = 1
        vis[fp] = 2
        vis[fn] = 3

        tag = f"{frame_idx:06d}"
        plt.imsave(os.path.join(self.viz_dir, f"gt_{tag}.png"), (g.astype(np.uint8) * 255), cmap="gray")
        plt.imsave(os.path.join(self.viz_dir, f"pred_{tag}.png"), (p.astype(np.uint8) * 255), cmap="gray")
        plt.imsave(os.path.join(self.viz_dir, f"omega_{tag}.png"), (valid.astype(np.uint8) * 255), cmap="gray")

        fig_path = os.path.join(self.viz_dir, f"viz_{tag}.png")
        cmap = ListedColormap(["black", "blue", "lime", "red"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        plt.figure(figsize=(6, 4))
        plt.imshow(vis, cmap=cmap, norm=norm, interpolation="nearest")
        plt.title("0:bg, 1:TP, 2:FP, 3:FN (on Omega)")
        plt.axis("off")
        plt.colorbar(ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    def destroy_node(self):
        if self.sync_policy == "strict_stamp" and self.write_missing_tail_rows:
            # frame_idx と stamp が対応付いたが評価できなかった行
            for f, skey in sorted(self.frame_idx_to_stamp.items(), key=lambda kv: kv[0]):
                if f in self.done:
                    continue
                row = [
                    int(f),                                  # frame_idx
                    int(self.stamp_points.get(skey, -1)),    # points_count
                    -1, -1, float("nan"), -1, float("nan"),
                    -1, float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"),
                    -1.0, -1.0, float("nan"),
                    1, 1,
                ]
                try:
                    self._wr.writerow(row)
                    self._ack(f)
                    self.done.add(f)
                except Exception:
                    pass

            # frame_idx だけ来て stamp 未対応の行
            while len(self.frame_queue) > 0:
                f = int(self.frame_queue.popleft())
                if f in self.done:
                    continue
                row = [
                    f, -1, -1, -1, float("nan"), -1, float("nan"),
                    -1, float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"),
                    -1.0, -1.0, float("nan"),
                    1, 1,
                ]
                try:
                    self._wr.writerow(row)
                    self._ack(f)
                    self.done.add(f)
                except Exception:
                    pass
        elif self.sequential_mode and self.write_missing_tail_rows and len(self.frame_queue) > 0:
            # frame_idx は来たが pred/omega が揃わなかった末尾分を欠損として残す
            while len(self.frame_queue) > 0:
                f = int(self.frame_queue.popleft())
                if f in self.done:
                    continue
                row = [
                    f, -1, -1, -1, float("nan"), -1, float("nan"),
                    -1, float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"),
                    -1.0, -1.0, float("nan"),
                    1, 1,
                ]
                try:
                    self._wr.writerow(row)
                    self._ack(f)
                    self.done.add(f)
                except Exception:
                    pass
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = RoiCoverLogger()
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
