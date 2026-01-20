# kitti_roi_eval/roi_eval_iou.py
from __future__ import annotations

import os
import csv
from typing import Dict, Tuple, Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

def _to_py(v):
    # numpy scalar 対策
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    return v


def make_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def stamp_key(msg: Image) -> Tuple[int, int]:
    return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))


def img_to_bool(msg: Image, expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    # mono8想定（>0 を True）
    if msg.encoding != "mono8":
        raise RuntimeError(f"Expected mono8, got {msg.encoding}")
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
    if expected_shape is not None and (a.shape != expected_shape):
        raise RuntimeError(f"Shape mismatch: got {a.shape}, expected {expected_shape}")
    return (a > 0)


class RoiIoUEvaluator(Node):
    """
    pred_topic: 予測マスク（重要候補領域など）  mono8
    gt_topic:   GTマスク（KITTI tracklet 投影など） mono8
    omega_topic: Valid(評価対象) マスク（Omega） mono8

    同一stampで pred/gt/omega が揃ったら，omega上で IoU/PRF を算出してCSVへ記録する．
    """

    def __init__(self):
        super().__init__("roi_eval_iou")

        # --- topics ---
        self.declare_parameter("pred_topic", "roi_est/roi_imp_mono8")
        self.declare_parameter("gt_topic", "pc_perturber/gt_mask_mono8")
        self.declare_parameter("omega_topic", "roi_est/omega_mono8")

        # --- outputs ---
        self.declare_parameter("out_dir", "")
        self.declare_parameter("csv_name", "iou_per_frame.csv")
        self.declare_parameter("warmup_sec", 5.0)

        # --- viz ---
        self.declare_parameter("viz_enable", True)
        self.declare_parameter("viz_max_frames", 10)

        # --- sync cache ---
        self.declare_parameter("cache_max_entries", 300)

        # ------------------------
        # read params
        # ------------------------
        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.omega_topic = str(self.get_parameter("omega_topic").value)

        out_dir = str(self.get_parameter("out_dir").value).strip()
        if out_dir == "":
            out_dir = os.path.abspath("result_kitti_iou")
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, str(self.get_parameter("csv_name").value))
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)

        self.viz_enable = bool(self.get_parameter("viz_enable").value)
        self.viz_max = int(self.get_parameter("viz_max_frames").value)
        self.viz_dir = os.path.join(self.out_dir, "viz")
        if self.viz_enable:
            os.makedirs(self.viz_dir, exist_ok=True)

        self.cache_max_entries = int(self.get_parameter("cache_max_entries").value)

        # cache[stamp] = {"pred":mask, "gt":mask, "omega":mask}
        self.cache: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self.first_stamp: Optional[Tuple[int, int]] = None
        self.expected_shape: Optional[Tuple[int, int]] = None

        self.frame_count = 0
        self.rows: List[List[float]] = []

        # CSV open
        self._fp = open(self.csv_path, "w", newline="")
        self._wr = csv.writer(self._fp)
        self._wr.writerow([
            "frame_idx",
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
        ])
        self._fp.flush()

        qos = make_qos(10)
        self.sub_pred = self.create_subscription(Image, self.pred_topic, self._cb_pred, qos)
        self.sub_gt = self.create_subscription(Image, self.gt_topic, self._cb_gt, qos)
        self.sub_omega = self.create_subscription(Image, self.omega_topic, self._cb_omega, qos)

        self.get_logger().info(
            "RoiIoUEvaluator started.\n"
            f"  pred={self.pred_topic}\n"
            f"  gt={self.gt_topic}\n"
            f"  omega(valid)={self.omega_topic}\n"
            f"  out_dir={self.out_dir}\n"
            f"  csv={self.csv_path}\n"
            f"  warmup_sec={self.warmup_sec}\n"
            f"  cache_max_entries={self.cache_max_entries}\n"
        )


    # ----------------------------
    # Callbacks
    # ----------------------------
    def _cb_pred(self, msg: Image):
        self._store("pred", msg)

    def _cb_gt(self, msg: Image):
        self._store("gt", msg)

    def _cb_omega(self, msg: Image):
        self._store("omega", msg)

    # ----------------------------
    # Sync store + eval
    # ----------------------------
    def _elapsed(self, key: Tuple[int, int]) -> float:
        if self.first_stamp is None:
            return 0.0
        s0, n0 = self.first_stamp
        s1, n1 = key
        return (s1 - s0) + (n1 - n0) * 1e-9

    def _prune_cache(self):
        # dict insertion order を利用して古いものから落とす
        while len(self.cache) > max(10, self.cache_max_entries):
            oldest = next(iter(self.cache))
            self.cache.pop(oldest, None)

    def _store(self, kind: str, msg: Image):
        key = stamp_key(msg)
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

        # 揃ったら評価
        if ("pred" in d) and ("gt" in d) and ("omega" in d):
            self._evaluate_one(key, d["pred"], d["gt"], d["omega"])
            self.cache.pop(key, None)

    def _evaluate_one(self, key: Tuple[int, int], pred: np.ndarray, gt: np.ndarray, omega: np.ndarray):
        # warmup除外
        if self._elapsed(key) < self.warmup_sec:
            return

        valid = omega  # Valid = Omega
        p = pred & valid
        g = gt & valid

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

        row = [
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
        ]
        self._wr.writerow(row)
        self._fp.flush()

        self.rows.append(row)
        self.frame_count += 1

        if self.viz_enable and self.frame_count <= self.viz_max:
            self._save_viz(self.frame_count, pred, gt, omega)

        if (self.frame_count % 10) == 0:
            self.get_logger().info(
                f"[IoU] frame={self.frame_count} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} "
                f"(gt={gt_bins}, pred={pred_bins}, omega={omega_bins}, empty={int(empty)})"
            )

    # ----------------------------
    # Viz
    # ----------------------------
    def _save_viz(self, idx: int, pred: np.ndarray, gt: np.ndarray, omega: np.ndarray):
        valid = omega.astype(bool)
        pred = pred.astype(bool)
        gt   = gt.astype(bool)

        p = pred & valid
        g = gt & valid
        tp = p & g
        fp = p & (~g)
        fn = (~p) & g

        # 0:bg, 1:TP, 2:FP, 3:FN
        vis = np.zeros_like(pred, dtype=np.uint8)
        vis[tp] = 1
        vis[fp] = 2
        vis[fn] = 3

        # --- save raw masks (0/255) ---
        gt_path    = os.path.join(self.viz_dir, f"gt_{idx:06d}.png")
        pred_path  = os.path.join(self.viz_dir, f"pred_{idx:06d}.png")
        omega_path = os.path.join(self.viz_dir, f"omega_{idx:06d}.png")

        plt.imsave(gt_path,   (g.astype(np.uint8) * 255), cmap="gray")
        plt.imsave(pred_path, (p.astype(np.uint8) * 255), cmap="gray")
        plt.imsave(omega_path,(valid.astype(np.uint8) * 255), cmap="gray")

        # --- save TP/FP/FN with fixed colors + colorbar ---
        fig_path = os.path.join(self.viz_dir, f"viz_{idx:06d}.png")
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


    # ----------------------------
    # Summary
    # ----------------------------
    def destroy_node(self):
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
        arr = np.array(self.rows, dtype=np.float64)  # columns aligned with row
        # col idx:
        # 0 frame_idx, 1 sec, 2 nsec, 3 iou, 4 prec, 5 rec, 6 f1, 7 gt, 8 pred, 9 omega, 10 empty
        frame_idx = arr[:, 0]
        iou = arr[:, 3]
        prec = arr[:, 4]
        rec = arr[:, 5]
        f1 = arr[:, 6]
        empty = arr[:, 10]

        def _p95(x: np.ndarray) -> float:
            return float(np.percentile(x, 95))

        summary_path = os.path.join(self.out_dir, "iou_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "median", "min", "max", "p95"])
            for name, x in [("iou", iou), ("precision", prec), ("recall", rec), ("f1", f1)]:
                w.writerow([
                    name,
                    float(np.mean(x)),
                    float(np.median(x)),
                    float(np.min(x)),
                    float(np.max(x)),
                    _p95(x),
                ])
            w.writerow([])
            w.writerow(["empty_union0_ratio", float(np.mean(empty))])

        # timeseries
        fig1 = os.path.join(self.out_dir, "iou_prf_timeseries.png")
        plt.figure(figsize=(8, 4))
        plt.plot(frame_idx, iou, label="IoU")
        plt.plot(frame_idx, prec, label="Precision")
        plt.plot(frame_idx, rec, label="Recall")
        plt.plot(frame_idx, f1, label="F1")
        plt.xlabel("frame_idx (after warmup)")
        plt.ylabel("score")
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig1, dpi=200)
        plt.close()

        # hist
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
