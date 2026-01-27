#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/roi_cover_logger.py

from __future__ import annotations

import os
import csv
import time
from typing import Optional, Dict, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Int32
from sensor_msgs.msg import Image


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
        self.declare_parameter("omega_topic", "roi_est/omega_mono8")
        self.declare_parameter("frame_idx_topic", "kitti_player/frame_idx")
        self.declare_parameter("points_count_topic", "kitti_player/points_count")
        self.declare_parameter("ack_topic", "kitti_player/ack_frame_idx")

        self.declare_parameter("gt_npz", "")
        self.declare_parameter("cover_tol_bins", 0)

        self.declare_parameter("out_dir", "")
        self.declare_parameter("csv_name", "cover_per_frame.csv")

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
        self.omega_topic = str(self.get_parameter("omega_topic").value)
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
        self.sub_pred = self.create_subscription(Image, self.pred_topic, self._cb_pred, qos_roi)
        self.sub_omega = self.create_subscription(Image, self.omega_topic, self._cb_omega, qos_roi)
        self.pub_ack = self.create_publisher(Int32, self.ack_topic, qos_ack)

        # -------------------------
        # state
        # -------------------------
        self.cur_frame: Optional[int] = None
        self.t_frame: Optional[float] = None
        self.points_count: Optional[int] = None

        self.pred: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
        self.t_pred: Optional[float] = None
        self.t_omega: Optional[float] = None

        self.done = set()

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
            f"  omega={self.omega_topic}\n"
            f"  frame_idx={self.frame_idx_topic}\n"
            f"  points_count={self.points_count_topic}\n"
            f"  ack={self.ack_topic}\n"
            f"  gt_npz={self.gt_npz} (frames={len(self.gt_frames)}, VxH={self.V}x{self.H})\n"
            f"  cover_tol_bins={self.cover_tol_bins}\n"
            f"  out_dir={self.out_dir}\n"
            f"  csv={self.csv_path}\n"
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
        self.omega = None
        self.t_pred = None
        self.t_omega = None

    def _cb_frame_idx(self, msg: Int32):
        f = int(msg.data)

        if f in self.done:
            # ACKロスト対策
            self._ack(f)
            return

        if self.cur_frame is None or f != self.cur_frame:
            self._reset_for_frame(f)

    def _cb_points_count(self, msg: Int32):
        if self.cur_frame is None:
            return
        self.points_count = int(msg.data)

    def _cb_pred(self, msg: Image):
        if self.cur_frame is None:
            return
        try:
            self.pred = img_to_bool_mono8(msg, self.expected_shape)
            self.t_pred = time.time()
        except Exception as e:
            self.get_logger().error(f"failed to parse pred: {e}")
            return
        self._try_eval()

    def _cb_omega(self, msg: Image):
        if self.cur_frame is None:
            return
        try:
            self.omega = img_to_bool_mono8(msg, self.expected_shape)
            self.t_omega = time.time()
        except Exception as e:
            self.get_logger().error(f"failed to parse omega: {e}")
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

        t0 = time.time()

        valid = self.omega
        p = self.pred & valid

        gi = self.gt_map.get(f, None)
        gt_missing = 0
        if gi is None:
            gt_missing = 1
            g = np.zeros_like(valid, dtype=bool)
        else:
            g = (self.gt_stack[gi] > 0) & valid

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

        lat_pred_ms = -1.0 if self.t_pred is None else (self.t_pred - self.t_frame) * 1000.0
        lat_omega_ms = -1.0 if self.t_omega is None else (self.t_omega - self.t_frame) * 1000.0
        eval_ms = (time.time() - t0) * 1000.0

        row = [
            f,
            int(self.points_count) if self.points_count is not None else -1,
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

        self.done.add(f)
        self._ack(f)

        if (len(self.done) % 20) == 0:
            self.get_logger().info(
                f"[cover] done={len(self.done)} frame={f} "
                f"cover={gt_cover_ratio:.3f} (tol={gt_cover_ratio_tol:.3f}) "
                f"pred_ratio={pred_ratio:.3f} gt_ratio={gt_ratio:.3f}"
            )

    def destroy_node(self):
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
