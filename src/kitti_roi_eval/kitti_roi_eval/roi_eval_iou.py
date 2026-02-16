#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/roi_eval_iou.py

from __future__ import annotations

import os
import csv
import time
from collections import deque
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm


def make_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def stamp_key_from_header(h) -> Tuple[int, int]:
    return (int(h.stamp.sec), int(h.stamp.nanosec))


def stamp_key_img(msg: Image) -> Tuple[int, int]:
    return stamp_key_from_header(msg.header)


def stamp_key_pc(msg: PointCloud2) -> Tuple[int, int]:
    return stamp_key_from_header(msg.header)


def img_to_bool(msg: Image, expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if msg.encoding != "mono8":
        raise RuntimeError(f"Expected mono8, got {msg.encoding}")
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
    if expected_shape is not None and (a.shape != expected_shape):
        raise RuntimeError(f"Shape mismatch: got {a.shape}, expected {expected_shape}")
    return (a > 0)


def pc2_num_points(msg: PointCloud2) -> int:
    # KITTI raw は通常 height=1, width=N だが、確実な式は data/point_step
    ps = int(msg.point_step)
    if ps <= 0:
        return int(msg.width) * int(msg.height)
    n = int(len(msg.data) // ps)
    return n


class RoiIoUEvaluator(Node):
    """
    pred_topic: 予測マスク mono8
    gt_topic:   GTマスク mono8
    omega_topic: Valid(Omega) mono8
    pc_topic:   点群(PointCloud2) 点数取得用

    同一stampで pred/gt/omega が揃ったら，omega上で IoU/PRF を算出してCSVへ記録する．
    """

    def __init__(self):
        super().__init__("roi_eval_iou")

        self.declare_parameter("pred_topic", "roi_est/roi_imp_mono8")
        self.declare_parameter("gt_topic", "pc_perturber/gt_mask_mono8")
        self.declare_parameter("omega_topic", "roi_est/omega_mono8")
        self.declare_parameter("pc_topic", "/livox/lidar_perturbed")

        self.declare_parameter("out_dir", "")
        self.declare_parameter("csv_name", "iou_per_frame.csv")
        self.declare_parameter("warmup_sec", 5.0)
        self.declare_parameter("sync_policy", "stamp")  # stamp | strict_stamp
        self.declare_parameter("frame_idx_mode", "after_warmup")  # after_warmup | topic
        self.declare_parameter("frame_idx_topic", "kitti_player/frame_idx")
        self.declare_parameter("sequential_mode", False)  # frame_idx を到着順に1対1対応
        self.declare_parameter("write_missing_tail_rows", True)
        self.declare_parameter("qos_meta_depth", 50)
        self.declare_parameter("qos_meta_reliable", False)

        self.declare_parameter("viz_enable", True)
        self.declare_parameter("viz_max_frames", 100000)

        self.declare_parameter("cache_max_entries", 300)

        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.omega_topic = str(self.get_parameter("omega_topic").value)
        self.pc_topic = str(self.get_parameter("pc_topic").value)

        out_dir = str(self.get_parameter("out_dir").value).strip()
        if out_dir == "":
            out_dir = os.path.abspath("result_kitti_iou")
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, str(self.get_parameter("csv_name").value))
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)
        self.sync_policy = str(self.get_parameter("sync_policy").value).strip().lower()
        if self.sync_policy not in ("stamp", "strict_stamp"):
            self.get_logger().warn(f"Unknown sync_policy='{self.sync_policy}', fallback to 'stamp'.")
            self.sync_policy = "stamp"
        self.frame_idx_mode = str(self.get_parameter("frame_idx_mode").value).strip().lower()
        if self.frame_idx_mode not in ("after_warmup", "topic"):
            self.get_logger().warn(
                f"Unknown frame_idx_mode='{self.frame_idx_mode}', fallback to 'after_warmup'."
            )
            self.frame_idx_mode = "after_warmup"
        self.frame_idx_topic = str(self.get_parameter("frame_idx_topic").value)
        self.sequential_mode = bool(self.get_parameter("sequential_mode").value)
        self.write_missing_tail_rows = bool(self.get_parameter("write_missing_tail_rows").value)
        if self.sync_policy == "strict_stamp":
            self.frame_idx_mode = "topic"
            if self.sequential_mode:
                self.get_logger().warn(
                    "sync_policy=strict_stamp ignores sequential_mode. Force sequential_mode=False."
                )
            self.sequential_mode = False

        self.viz_enable = bool(self.get_parameter("viz_enable").value)
        self.viz_max = int(self.get_parameter("viz_max_frames").value)
        self.viz_dir = os.path.join(self.out_dir, "viz")
        if self.viz_enable:
            os.makedirs(self.viz_dir, exist_ok=True)

        self.cache_max_entries = int(self.get_parameter("cache_max_entries").value)

        # cache[stamp] = {"pred":bool, "gt":bool, "omega":bool, "n_points":int}
        self.cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.first_stamp: Optional[Tuple[int, int]] = None
        self.expected_shape: Optional[Tuple[int, int]] = None

        self.frame_count = 0
        self.rows: List[List[float]] = []
        self._latest_frame_idx: Optional[int] = None
        self._frame_idx_queue: deque[int] = deque()
        self._pc_points_queue: deque[int] = deque()
        self._pc_stamp_queue: deque[Tuple[int, int]] = deque()
        self._stamp_to_frame_idx: Dict[Tuple[int, int], int] = {}
        self._frame_idx_to_stamp: Dict[int, Tuple[int, int]] = {}
        self._stamp_points: Dict[Tuple[int, int], int] = {}
        self._evaluated_frame_idx: set[int] = set()

        # CSV open
        self._fp = open(self.csv_path, "w", newline="")
        self._wr = csv.writer(self._fp)
        self._wr.writerow([
            "frame_idx",
            "frame_idx_after_warmup",
            "stamp_sec",
            "stamp_nanosec",
            "iou",
            "precision",
            "recall",
            "f1",
            "gt_bins",
            "pred_bins",
            "omega_bins",
            "empty_union0",
            # --- added ---
            "proc_time_ms",
            "gt_cover_ratio",
            "points_in_frame",
            "total_bins",
            "missing_data",
        ])
        self._fp.flush()

        qos = make_qos(10)
        qos_meta = make_qos(
            int(self.get_parameter("qos_meta_depth").value)
        ) if not bool(self.get_parameter("qos_meta_reliable").value) else QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(max(1, int(self.get_parameter("qos_meta_depth").value))),
        )
        self.sub_pred = self.create_subscription(Image, self.pred_topic, self._cb_pred, qos)
        self.sub_gt = self.create_subscription(Image, self.gt_topic, self._cb_gt, qos)
        self.sub_omega = self.create_subscription(Image, self.omega_topic, self._cb_omega, qos)
        self.sub_pc = self.create_subscription(PointCloud2, self.pc_topic, self._cb_pc, qos)
        self.sub_frame_idx = None
        if self.frame_idx_mode == "topic":
            self.sub_frame_idx = self.create_subscription(
                Int32, self.frame_idx_topic, self._cb_frame_idx, qos_meta
            )

        self.get_logger().info(
            "RoiIoUEvaluator started.\n"
            f"  pred={self.pred_topic}\n"
            f"  gt={self.gt_topic}\n"
            f"  omega(valid)={self.omega_topic}\n"
            f"  pc_topic={self.pc_topic}\n"
            f"  out_dir={self.out_dir}\n"
            f"  csv={self.csv_path}\n"
            f"  warmup_sec={self.warmup_sec}\n"
            f"  sync_policy={self.sync_policy}\n"
            f"  frame_idx_mode={self.frame_idx_mode}\n"
            f"  frame_idx_topic={self.frame_idx_topic}\n"
            f"  sequential_mode={self.sequential_mode}\n"
            f"  cache_max_entries={self.cache_max_entries}\n"
        )

    def _cb_frame_idx(self, msg: Int32):
        self._latest_frame_idx = int(msg.data)
        if self.sync_policy == "strict_stamp":
            self._frame_idx_queue.append(int(msg.data))
            paired_keys = self._pair_frameidx_with_pc_stamp()
            for key in paired_keys:
                d = self.cache.get(key)
                if d is not None and ("pred" in d) and ("gt" in d) and ("omega" in d):
                    self._evaluate_one(
                        key,
                        d["pred"],
                        d["gt"],
                        d["omega"],
                        int(d.get("n_points", self._stamp_points.get(key, -1))),
                    )
                    self.cache.pop(key, None)
            return
        if self.frame_idx_mode == "topic" and self.sequential_mode:
            self._frame_idx_queue.append(int(msg.data))

    def _cb_pred(self, msg: Image):
        self._store_img("pred", msg)

    def _cb_gt(self, msg: Image):
        self._store_img("gt", msg)

    def _cb_omega(self, msg: Image):
        self._store_img("omega", msg)

    def _cb_pc(self, msg: PointCloud2):
        key = stamp_key_pc(msg)
        if self.first_stamp is None:
            self.first_stamp = key

        n = pc2_num_points(msg)
        if self.frame_idx_mode == "topic" and self.sequential_mode:
            self._pc_points_queue.append(int(n))
        self._stamp_points[key] = int(n)
        if self.sync_policy == "strict_stamp":
            self._pc_stamp_queue.append(key)
            paired_keys = self._pair_frameidx_with_pc_stamp()
            for pkey in paired_keys:
                d2 = self.cache.get(pkey)
                if d2 is not None and ("pred" in d2) and ("gt" in d2) and ("omega" in d2):
                    self._evaluate_one(
                        pkey,
                        d2["pred"],
                        d2["gt"],
                        d2["omega"],
                        int(d2.get("n_points", self._stamp_points.get(pkey, -1))),
                    )
                    self.cache.pop(pkey, None)
        d = self.cache.get(key)
        if d is None:
            d = {}
            self.cache[key] = d
        d["n_points"] = int(n)

        self._prune_cache()

    def _elapsed(self, key: Tuple[int, int]) -> float:
        if self.first_stamp is None:
            return 0.0
        s0, n0 = self.first_stamp
        s1, n1 = key
        return (s1 - s0) + (n1 - n0) * 1e-9

    def _prune_cache(self):
        while len(self.cache) > max(10, self.cache_max_entries):
            oldest = next(iter(self.cache))
            self.cache.pop(oldest, None)

    def _pair_frameidx_with_pc_stamp(self) -> List[Tuple[int, int]]:
        paired: List[Tuple[int, int]] = []
        while len(self._frame_idx_queue) > 0 and len(self._pc_stamp_queue) > 0:
            fidx = int(self._frame_idx_queue.popleft())
            skey = self._pc_stamp_queue.popleft()
            self._stamp_to_frame_idx[skey] = fidx
            self._frame_idx_to_stamp[fidx] = skey
            paired.append(skey)
        return paired

    def _store_img(self, kind: str, msg: Image):
        key = stamp_key_img(msg)
        if self.first_stamp is None:
            self.first_stamp = key

        if self.expected_shape is None:
            self.expected_shape = (int(msg.height), int(msg.width))

        try:
            mask = img_to_bool(msg, expected_shape=self.expected_shape)
        except Exception as e:
            self.get_logger().error(f"failed to parse {kind} mask: {e}")
            return

        d = self.cache.get(key)
        if d is None:
            d = {}
            self.cache[key] = d
        d[kind] = mask

        self._prune_cache()

        if ("pred" in d) and ("gt" in d) and ("omega" in d):
            n_points = int(d.get("n_points", -1))
            self._evaluate_one(key, d["pred"], d["gt"], d["omega"], n_points)
            self.cache.pop(key, None)

    def _evaluate_one(
        self,
        key: Tuple[int, int],
        pred: np.ndarray,
        gt: np.ndarray,
        omega: np.ndarray,
        n_points: int,
    ):
        if self._elapsed(key) < self.warmup_sec:
            return
        if self.sync_policy == "strict_stamp":
            if key not in self._stamp_to_frame_idx:
                return
            frame_idx_out = int(self._stamp_to_frame_idx[key])
            if frame_idx_out in self._evaluated_frame_idx:
                return
            n_points = int(self._stamp_points.get(key, n_points))
        if self.frame_idx_mode == "topic" and (not self.sequential_mode) and self._latest_frame_idx is None:
            # frame_idx がまだ来ていない間は記録しない（cover_logger と同様に frame_idx 主導）
            if self.sync_policy != "strict_stamp":
                return

        t0 = time.perf_counter()

        valid = omega.astype(bool)
        pred_b = pred.astype(bool)
        gt_b = gt.astype(bool)

        p = pred_b & valid
        g = gt_b & valid

        tp = int(np.sum(p & g))
        fp = int(np.sum(p & (~g)))
        fn = int(np.sum((~p) & g))

        union = tp + fp + fn
        empty = (union == 0)

        if empty:
            iou = 1.0
            prec = 1.0
            rec = 1.0
            f1 = 1.0
        else:
            eps = 1e-9
            iou = tp / (union + eps)
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f1 = (2.0 * prec * rec) / (prec + rec + eps)

        gt_bins = int(np.sum(g))
        pred_bins = int(np.sum(p))
        omega_bins = int(np.sum(valid))
        total_bins = int(pred.size)

        eps = 1e-9
        gt_cover_ratio = float(tp) / float(gt_bins + eps)

        t1 = time.perf_counter()
        proc_time_ms = (t1 - t0) * 1e3

        if self.sync_policy == "strict_stamp":
            pass
        elif self.frame_idx_mode == "topic":
            if self.sequential_mode:
                if len(self._frame_idx_queue) == 0:
                    return
                frame_idx_out = int(self._frame_idx_queue.popleft())
            else:
                frame_idx_out = int(self._latest_frame_idx) if self._latest_frame_idx is not None else int(self.frame_count)
        else:
            frame_idx_out = int(self.frame_count)

        if self.frame_idx_mode == "topic" and self.sequential_mode and len(self._pc_points_queue) > 0:
            n_points = int(self._pc_points_queue.popleft())

        row = [
            frame_idx_out,
            int(self.frame_count),
            int(key[0]),
            int(key[1]),
            float(iou),
            float(prec),
            float(rec),
            float(f1),
            int(gt_bins),
            int(pred_bins),
            int(omega_bins),
            int(empty),
            # --- added ---
            float(proc_time_ms),
            float(gt_cover_ratio),
            int(n_points),
            int(total_bins),
            0,
        ]
        self._wr.writerow(row)
        self._fp.flush()

        self.rows.append(row)
        self.frame_count += 1
        if self.sync_policy == "strict_stamp":
            self._evaluated_frame_idx.add(int(frame_idx_out))

        if self.viz_enable and self.frame_count <= self.viz_max:
            self._save_viz(frame_idx_out, pred_b, gt_b, valid)

        if (self.frame_count % 10) == 0:
            self.get_logger().info(
                f"[IoU] frame={self.frame_count} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} "
                f"(gt={gt_bins}, pred={pred_bins}, omega={omega_bins}, pts={n_points}, ms={proc_time_ms:.2f})"
            )

    def _save_viz(self, idx: int, pred: np.ndarray, gt: np.ndarray, omega: np.ndarray):
        valid = omega.astype(bool)
        pred = pred.astype(bool)
        gt = gt.astype(bool)

        p = pred & valid
        g = gt & valid
        tp = p & g
        fp = p & (~g)
        fn = (~p) & g

        vis = np.zeros_like(pred, dtype=np.uint8)
        vis[tp] = 1
        vis[fp] = 2
        vis[fn] = 3

        gt_path = os.path.join(self.viz_dir, f"gt_{idx:06d}.png")
        pred_path = os.path.join(self.viz_dir, f"pred_{idx:06d}.png")
        omega_path = os.path.join(self.viz_dir, f"omega_{idx:06d}.png")

        plt.imsave(gt_path, (g.astype(np.uint8) * 255), cmap="gray")
        plt.imsave(pred_path, (p.astype(np.uint8) * 255), cmap="gray")
        plt.imsave(omega_path, (valid.astype(np.uint8) * 255), cmap="gray")

        fig_path = os.path.join(self.viz_dir, f"viz_{idx:06d}.png")
        cmap = ListedColormap(["black", "blue", "lime", "red"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        plt.figure(figsize=(6, 4))
        plt.imshow(vis, cmap=cmap, norm=norm, interpolation="nearest")
        plt.title("0:bg, 1:TP, 2:FP, 3:FN")
        plt.axis("off")
        plt.colorbar(ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    def destroy_node(self):
        if self.sync_policy == "strict_stamp" and self.write_missing_tail_rows:
            for fidx, skey in sorted(self._frame_idx_to_stamp.items(), key=lambda kv: kv[0]):
                if int(fidx) in self._evaluated_frame_idx:
                    continue
                row = [
                    int(fidx),             # frame_idx
                    int(self.frame_count), # frame_idx_after_warmup
                    int(skey[0]),          # stamp_sec
                    int(skey[1]),          # stamp_nanosec
                    float("nan"),          # iou
                    float("nan"),          # precision
                    float("nan"),          # recall
                    float("nan"),          # f1
                    -1,                    # gt_bins
                    -1,                    # pred_bins
                    -1,                    # omega_bins
                    1,                     # empty_union0
                    float("nan"),          # proc_time_ms
                    float("nan"),          # gt_cover_ratio
                    int(self._stamp_points.get(skey, -1)),  # points
                    -1,                    # total_bins
                    1,                     # missing_data
                ]
                try:
                    self._wr.writerow(row)
                    self.rows.append(row)
                    self.frame_count += 1
                except Exception:
                    pass
            while len(self._frame_idx_queue) > 0:
                fidx = int(self._frame_idx_queue.popleft())
                row = [
                    fidx,                  # frame_idx
                    int(self.frame_count), # frame_idx_after_warmup
                    -1,                    # stamp_sec
                    -1,                    # stamp_nanosec
                    float("nan"),          # iou
                    float("nan"),          # precision
                    float("nan"),          # recall
                    float("nan"),          # f1
                    -1,                    # gt_bins
                    -1,                    # pred_bins
                    -1,                    # omega_bins
                    1,                     # empty_union0
                    float("nan"),          # proc_time_ms
                    float("nan"),          # gt_cover_ratio
                    -1,                    # points
                    -1,                    # total_bins
                    1,                     # missing_data
                ]
                try:
                    self._wr.writerow(row)
                    self.rows.append(row)
                    self.frame_count += 1
                except Exception:
                    pass
        elif self.frame_idx_mode == "topic" and self.sequential_mode and self.write_missing_tail_rows:
            # frame_idx は来たが pred/gt/omega が揃わなかった末尾分を欠損行として残す
            while len(self._frame_idx_queue) > 0:
                fidx = int(self._frame_idx_queue.popleft())
                row = [
                    fidx,                  # frame_idx
                    int(self.frame_count), # frame_idx_after_warmup
                    -1,                    # stamp_sec
                    -1,                    # stamp_nanosec
                    float("nan"),          # iou
                    float("nan"),          # precision
                    float("nan"),          # recall
                    float("nan"),          # f1
                    -1,                    # gt_bins
                    -1,                    # pred_bins
                    -1,                    # omega_bins
                    1,                     # empty_union0
                    float("nan"),          # proc_time_ms
                    float("nan"),          # gt_cover_ratio
                    int(self._pc_points_queue.popleft()) if len(self._pc_points_queue) > 0 else -1,  # points
                    -1,                    # total_bins
                    1,                     # missing_data
                ]
                try:
                    self._wr.writerow(row)
                    self.rows.append(row)
                    self.frame_count += 1
                except Exception:
                    pass

        try:
            if self._fp is not None:
                try:
                    self._fp.flush()
                except Exception:
                    pass
                try:
                    self._fp.close()
                except Exception:
                    pass
        except Exception:
            pass

        if len(self.rows) > 0:
            self._write_summary_and_plots()

        return super().destroy_node()

    def _write_summary_and_plots(self):
        arr = np.array(self.rows, dtype=np.float64)
        if arr.shape[0] == 0:
            return
        if arr.shape[1] > 16:
            valid_rows = np.isfinite(arr[:, 16]) & (arr[:, 16] < 0.5)
        else:
            valid_rows = np.ones((arr.shape[0],), dtype=bool)
        if not np.any(valid_rows):
            self.get_logger().warn("No valid rows for summary/plot.")
            return
        arr = arr[valid_rows]
        frame_idx = arr[:, 0]
        iou = arr[:, 4]
        prec = arr[:, 5]
        rec = arr[:, 6]
        f1 = arr[:, 7]
        empty = arr[:, 11]

        def _p95(x: np.ndarray) -> float:
            return float(np.percentile(x, 95))

        summary_path = os.path.join(self.out_dir, "iou_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "median", "min", "max", "p95"])
            for name, x in [("iou", iou), ("precision", prec), ("recall", rec), ("f1", f1)]:
                w.writerow([name, float(np.mean(x)), float(np.median(x)), float(np.min(x)), float(np.max(x)), _p95(x)])
            w.writerow([])
            w.writerow(["empty_union0_ratio", float(np.mean(empty))])

        fig1 = os.path.join(self.out_dir, "iou_prf_timeseries.png")
        plt.figure(figsize=(8, 4))
        plt.plot(frame_idx, iou, label="IoU")
        plt.plot(frame_idx, prec, label="Precision")
        plt.plot(frame_idx, rec, label="Recall")
        plt.plot(frame_idx, f1, label="F1")
        plt.xlabel("frame_idx")
        plt.ylabel("score")
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig1, dpi=200)
        plt.close()

        fig2 = os.path.join(self.out_dir, "iou_hist.png")
        plt.figure(figsize=(6, 4))
        plt.hist(iou, bins=30)
        plt.xlabel("IoU")
        plt.ylabel("count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig2, dpi=200)
        plt.close()

        self.get_logger().info(f"[eval] wrote: {summary_path}")
        self.get_logger().info(f"[eval] wrote: {fig1}")
        self.get_logger().info(f"[eval] wrote: {fig2}")


def main():
    rclpy.init()
    node = RoiIoUEvaluator()
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
