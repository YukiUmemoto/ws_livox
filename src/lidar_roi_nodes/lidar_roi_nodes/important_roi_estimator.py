#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import os
import csv
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from sensor_msgs_py import point_cloud2

from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError


# --------------------------
# QoS helper
# --------------------------
def make_qos_profile(
    reliability: str = "best_effort",
    durability: str = "volatile",
    history: str = "keep_last",
    depth: int = 5,
) -> QoSProfile:
    rel = reliability.lower()
    dur = durability.lower()
    his = history.lower()

    if rel == "reliable":
        r = ReliabilityPolicy.RELIABLE
    elif rel == "best_effort":
        r = ReliabilityPolicy.BEST_EFFORT
    else:
        raise ValueError(f"Unknown reliability={reliability}")

    if dur == "volatile":
        d = DurabilityPolicy.VOLATILE
    elif dur in ("transient_local", "transient"):
        d = DurabilityPolicy.TRANSIENT_LOCAL
    else:
        raise ValueError(f"Unknown durability={durability}")

    if his == "keep_last":
        h = HistoryPolicy.KEEP_LAST
    elif his == "keep_all":
        h = HistoryPolicy.KEEP_ALL
    else:
        raise ValueError(f"Unknown history={history}")

    depth = int(max(1, depth))
    return QoSProfile(reliability=r, durability=d, history=h, depth=depth)


# --------------------------
# Stats / profiler
# --------------------------
@dataclass
class _Stat:
    n: int = 0
    ema_ms: float = 0.0
    max_ms: float = 0.0


class Profiler:
    def __init__(self, node: Node, enabled: bool, log_period_frames: int, warn_ms: float, ema_alpha: float = 0.1):
        self.node = node
        self.enabled = enabled
        self.log_period_frames = max(1, int(log_period_frames))
        self.warn_ms = float(warn_ms)
        self.ema_alpha = float(ema_alpha)
        self._frame_count = 0
        self._stats = {}
        self._this_frame = {}

    def frame_start(self):
        if not self.enabled:
            return
        self._this_frame.clear()

    def mark(self, name: str, t0_ns: int, t1_ns: int):
        if not self.enabled:
            return
        ms = (t1_ns - t0_ns) / 1e6
        self._this_frame[name] = ms
        st = self._stats.get(name)
        if st is None:
            st = _Stat(n=0, ema_ms=ms, max_ms=ms)
            self._stats[name] = st
        st.n += 1
        st.ema_ms = (1.0 - self.ema_alpha) * st.ema_ms + self.ema_alpha * ms
        st.max_ms = max(st.max_ms, ms)

    def frame_end(self):
        if not self.enabled:
            return
        self._frame_count += 1

        over = [(k, v) for k, v in self._this_frame.items() if v >= self.warn_ms]
        if over:
            over_str = ", ".join([f"{k}={v:.1f}ms" for k, v in sorted(over, key=lambda x: -x[1])])
            self.node.get_logger().warn(f"[PROFILE] slow sections: {over_str}")

        if (self._frame_count % self.log_period_frames) == 0:
            items = sorted(self._stats.items(), key=lambda kv: -kv[1].ema_ms)
            lines = [f"{name}: ema={st.ema_ms:.2f}ms, max={st.max_ms:.2f}ms (n={st.n})" for name, st in items[:60]]
            self.node.get_logger().info("[PROFILE] summary:\n  " + "\n  ".join(lines))


# --------------------------
# Estimator node (focused + CSV proc_time_ms)
# --------------------------
class ImportantROIWithReliability(Node):
    def __init__(self):
        super().__init__("important_roi_estimator")

        # ===== QoS params =====
        self.declare_parameter("sub_reliability", "best_effort")
        self.declare_parameter("sub_durability", "volatile")
        self.declare_parameter("sub_history", "keep_last")
        self.declare_parameter("sub_depth", 10)

        self.declare_parameter("pub_reliability", "best_effort")
        self.declare_parameter("pub_durability", "volatile")
        self.declare_parameter("pub_history", "keep_last")
        self.declare_parameter("pub_depth", 10)

        # ===== Topics =====
        self.declare_parameter("input_topic", "/livox/lidar_perturbed")

        # （互換のため宣言だけ残す）
        self.declare_parameter("gt_mask_topic", "pc_perturber/gt_mask_mono8")
        self.declare_parameter("drop_ratio_topic", "pc_perturber/drop_ratio")
        self.declare_parameter("bias_topic", "pc_perturber/bias_m")

        # ===== Binning =====
        self.declare_parameter("horizontal_fov_deg", 70.4)
        self.declare_parameter("vertical_fov_deg", 77.2)
        self.declare_parameter("num_horizontal_bins", 128)
        self.declare_parameter("num_vertical_bins", 128)

        # ===== Feature / importance =====
        self.declare_parameter("enable_temporal", True)
        self.declare_parameter("ema_alpha", 0.05)
        self.declare_parameter("min_expected_hits", 3.0)
        self.declare_parameter("N_min_compare", 1)

        self.declare_parameter("w_s", 0.5)
        self.declare_parameter("w_t", 0.5)
        self.declare_parameter("roi_top_percent", 10.0)

        # ===== Reliability (bin) =====
        self.declare_parameter("w_m", 0.4)
        self.declare_parameter("w_d", 0.3)
        self.declare_parameter("w_n", 0.3)
        self.declare_parameter("scale_d_m", 0.5)
        self.declare_parameter("n_floor", 1.0)
        self.declare_parameter("sigmoid_beta", 8.0)
        self.declare_parameter("sigmoid_center_c", 0.6)
        self.declare_parameter("tau_rel", 0.8)

        # ===== Outputs =====
        self.declare_parameter("publish_masked_maps", True)

        # ===== Profiling =====
        self.declare_parameter("profile_enable", True)
        self.declare_parameter("profile_log_period_frames", 30)
        self.declare_parameter("profile_warn_ms", 100.0)
        self.declare_parameter("profile_ema_alpha", 0.1)

        # ===== CSV logging (proc_time_ms 必須) =====
        self.declare_parameter("csv_enable", False)
        self.declare_parameter("csv_path", "")
        self.declare_parameter("csv_flush_every", 30)
        self.declare_parameter("csv_write_header", True)

        # ===== CSV logging (stats for Chap7) =====
        self.declare_parameter("stats_enable", False)
        self.declare_parameter("stats_csv_path", "")
        self.declare_parameter("stats_flush_every", 30)
        self.declare_parameter("stats_write_header", True)

        # ===== Optional outputs =====
        self.declare_parameter("publish_rel_low", False)

        # --------------------
        # Init params
        # --------------------
        self.input_topic = self.get_parameter("input_topic").value
        self.Hfov = float(self.get_parameter("horizontal_fov_deg").value)
        self.Vfov = float(self.get_parameter("vertical_fov_deg").value)
        self.H = int(self.get_parameter("num_horizontal_bins").value)
        self.V = int(self.get_parameter("num_vertical_bins").value)

        self.theta_min = math.radians(-self.Hfov / 2.0)
        self.theta_max = math.radians(+self.Hfov / 2.0)
        self.phi_min = math.radians(-self.Vfov / 2.0)
        self.phi_max = math.radians(+self.Vfov / 2.0)
        self.d_theta = (self.theta_max - self.theta_min) / self.H
        self.d_phi = (self.phi_max - self.phi_min) / self.V

        # EMA states
        self.expected_range = np.zeros((self.V, self.H), dtype=np.float32)
        self.expected_hits = np.zeros((self.V, self.H), dtype=np.float32)

        self._last_R_hat: Optional[np.ndarray] = None
        self._last_gt_mask: Optional[np.ndarray] = None
        self._last_drop_ratio: float = 0.0
        self._last_bias_m: float = 0.0

        self._frame_counter = 0

        # QoS
        sub_qos = make_qos_profile(
            self.get_parameter("sub_reliability").value,
            self.get_parameter("sub_durability").value,
            self.get_parameter("sub_history").value,
            int(self.get_parameter("sub_depth").value),
        )
        pub_qos = make_qos_profile(
            self.get_parameter("pub_reliability").value,
            self.get_parameter("pub_durability").value,
            self.get_parameter("pub_history").value,
            int(self.get_parameter("pub_depth").value),
        )

        # Subscriptions
        self.sub_pc = self.create_subscription(PointCloud2, self.input_topic, self.cb_pc, sub_qos)
        # 互換のため購読だけ残す（計測対象ではない）
        self.sub_gt = self.create_subscription(Image, self.get_parameter("gt_mask_topic").value, self.cb_gt, sub_qos)
        self.sub_drop = self.create_subscription(Float32, self.get_parameter("drop_ratio_topic").value, self.cb_drop, sub_qos)
        self.sub_bias = self.create_subscription(Float32, self.get_parameter("bias_topic").value, self.cb_bias, sub_qos)

        # Publishers（最小限）
        self.pub_I = self.create_publisher(Float32MultiArray, "roi_est/importance_map", pub_qos)
        self.pub_Rel = self.create_publisher(Float32MultiArray, "roi_est/rel_map", pub_qos)

        self.pub_roi_imp = self.create_publisher(Image, "roi_est/roi_imp_mono8", pub_qos)
        self.pub_roi_use = self.create_publisher(Image, "roi_est/roi_use_mono8", pub_qos)
        self.pub_roi_alert = self.create_publisher(Image, "roi_est/roi_alert_mono8", pub_qos)
        self.pub_omega = self.create_publisher(Image, "roi_est/omega_mono8", pub_qos)
        self.pub_rel_low = self.create_publisher(Image, "roi_est/rel_low_mono8", pub_qos)

        self.pub_frame_rel_all = self.create_publisher(Float32, "roi_est/frame_rel", pub_qos)
        self.pub_frame_rel_obs = self.create_publisher(Float32, "roi_est/frame_rel_obs", pub_qos)
        self.pub_alert_ratio = self.create_publisher(Float32, "roi_est/alert_ratio", pub_qos)
        self.pub_alert_ratio_omega = self.create_publisher(Float32, "roi_est/alert_ratio_omega", pub_qos)

        # Profiling
        self.prof = Profiler(
            node=self,
            enabled=bool(self.get_parameter("profile_enable").value),
            log_period_frames=int(self.get_parameter("profile_log_period_frames").value),
            warn_ms=float(self.get_parameter("profile_warn_ms").value),
            ema_alpha=float(self.get_parameter("profile_ema_alpha").value),
        )

        # CSV
        self._csv_enabled = bool(self.get_parameter("csv_enable").value)
        self._csv_fp = None
        self._csv_writer = None
        self._csv_flush_every = int(self.get_parameter("csv_flush_every").value)
        self._csv_rows_since_flush = 0
        if self._csv_enabled:
            self._init_csv_writer()

        # Stats CSV (Chap7)
        self._stats_enabled = bool(self.get_parameter("stats_enable").value)
        self._stats_fp = None
        self._stats_writer = None
        self._stats_flush_every = int(self.get_parameter("stats_flush_every").value)
        self._stats_rows_since_flush = 0
        if self._stats_enabled:
            self._init_stats_writer()

        self.get_logger().info(
            "ImportantROIWithReliability (focused+CSV) started.\n"
            f"  input={self.input_topic}\n"
            f"  bins=VxH={self.V}x{self.H}\n"
            f"  csv_enable={self._csv_enabled}\n"
            f"  stats_enable={self._stats_enabled}\n"
            "  publish: omega, roi_imp/use/alert, importance_map, rel_map, frame_rel(all/obs), alert_ratio(_omega)\n"
        )

    # --------------------
    # CSV helpers
    # --------------------
    def _csv_header(self):
        # 最小限（proc_time_ms を必ず含む）
        return [
            "frame_idx",
            "stamp_sec",
            "stamp_nanosec",
            "n_points",
            "proc_time_ms",
        ]

    def _init_csv_writer(self):
        path = str(self.get_parameter("csv_path").value).strip()
        if path == "":
            path = os.path.abspath("roi_est_proc_time.csv")
        path = os.path.expanduser(path)

        d = os.path.dirname(path)
        if d != "":
            os.makedirs(d, exist_ok=True)

        try:
            self._csv_fp = open(path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_fp)

            if bool(self.get_parameter("csv_write_header").value):
                try:
                    if os.path.getsize(path) == 0:
                        self._csv_writer.writerow(self._csv_header())
                        self._csv_fp.flush()
                except Exception:
                    pass

            self.get_logger().info(f"[CSV] logging enabled: {path}")
        except Exception as e:
            self.get_logger().error(f"[CSV] failed to open '{path}': {e}")
            self._csv_enabled = False
            self._csv_fp = None
            self._csv_writer = None

    def _csv_write_row(self, header, n_points: int, proc_time_ms: float):
        if (not self._csv_enabled) or (self._csv_writer is None):
            return

        try:
            sec = int(header.stamp.sec)
            nsec = int(header.stamp.nanosec)
        except Exception:
            sec, nsec = 0, 0

        row = [
            int(self._frame_counter),
            sec,
            nsec,
            int(n_points),
            float(proc_time_ms),  # ★列名proc_time_ms
        ]

        try:
            self._csv_writer.writerow(row)
            self._csv_rows_since_flush += 1
            if self._csv_rows_since_flush >= max(1, self._csv_flush_every):
                try:
                    self._csv_fp.flush()
                except Exception:
                    pass
                self._csv_rows_since_flush = 0
        except Exception as e:
            self.get_logger().error(f"[CSV] write failed: {e}")

    # --------------------
    # Stats CSV helpers (Chap7)
    # --------------------
    def _stats_header(self):
        return [
            "frame_idx",
            "stamp_sec",
            "stamp_nanosec",
            "n_points",
            "omega_bins",
            "roi_bins",
            "keepbin_ratio",
            "roi_top_percent",
            "roi_points",
            "keeppt_ratio",
            "S_mean_roi",
            "T_mean_roi",
            "I_mean_roi",
            "frame_rel_all",
            "frame_rel_obs",
            "alert_ratio",
            "alert_ratio_omega",
            "drop_ratio",
            "bias_m",
        ]

    def _init_stats_writer(self):
        path = str(self.get_parameter("stats_csv_path").value).strip()
        if path == "":
            path = os.path.abspath("roi_est_stats.csv")
        path = os.path.expanduser(path)

        d = os.path.dirname(path)
        if d != "":
            os.makedirs(d, exist_ok=True)

        try:
            self._stats_fp = open(path, "a", newline="")
            self._stats_writer = csv.writer(self._stats_fp)

            if bool(self.get_parameter("stats_write_header").value):
                try:
                    if os.path.getsize(path) == 0:
                        self._stats_writer.writerow(self._stats_header())
                        self._stats_fp.flush()
                except Exception:
                    pass

            self.get_logger().info(f"[STATS] logging enabled: {path}")
        except Exception as e:
            self.get_logger().error(f"[STATS] failed to open '{path}': {e}")
            self._stats_enabled = False
            self._stats_fp = None
            self._stats_writer = None

    def _stats_write_row(
        self,
        header,
        n_points: int,
        omega_bins: int,
        roi_bins: int,
        keepbin_ratio: float,
        roi_points: int,
        keeppt_ratio: float,
        s_mean_roi: float,
        t_mean_roi: float,
        i_mean_roi: float,
        frame_rel_all: float,
        frame_rel_obs: float,
        alert_ratio: float,
        alert_ratio_omega: float,
        drop_ratio: float,
        bias_m: float,
    ):
        if (not self._stats_enabled) or (self._stats_writer is None):
            return

        try:
            sec = int(header.stamp.sec)
            nsec = int(header.stamp.nanosec)
        except Exception:
            sec, nsec = 0, 0

        row = [
            int(self._frame_counter),
            sec,
            nsec,
            int(n_points),
            int(omega_bins),
            int(roi_bins),
            float(keepbin_ratio),
            float(self.get_parameter("roi_top_percent").value),
            int(roi_points),
            float(keeppt_ratio),
            float(s_mean_roi),
            float(t_mean_roi),
            float(i_mean_roi),
            float(frame_rel_all),
            float(frame_rel_obs),
            float(alert_ratio),
            float(alert_ratio_omega),
            float(drop_ratio),
            float(bias_m),
        ]

        try:
            self._stats_writer.writerow(row)
            self._stats_rows_since_flush += 1
            if self._stats_rows_since_flush >= max(1, self._stats_flush_every):
                try:
                    self._stats_fp.flush()
                except Exception:
                    pass
                self._stats_rows_since_flush = 0
        except Exception as e:
            self.get_logger().error(f"[STATS] write failed: {e}")

    # --------------------
    # GT callbacks (compat)
    # --------------------
    def cb_gt(self, msg: Image):
        if msg.encoding != "mono8":
            return
        if msg.height != self.V or msg.width != self.H:
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(self.V, self.H)
        self._last_gt_mask = (a > 0)

    def cb_drop(self, msg: Float32):
        self._last_drop_ratio = float(msg.data)

    def cb_bias(self, msg: Float32):
        self._last_bias_m = float(msg.data)

    # --------------------
    # Utilities: publish
    # --------------------
    def _publish_map(self, pub, arr2d: np.ndarray):
        msg = Float32MultiArray()
        dim_v = MultiArrayDimension()
        dim_v.label = "vertical"
        dim_v.size = self.V
        dim_v.stride = self.V * self.H
        dim_h = MultiArrayDimension()
        dim_h.label = "horizontal"
        dim_h.size = self.H
        dim_h.stride = self.H
        msg.layout.dim = [dim_v, dim_h]
        msg.layout.data_offset = 0
        msg.data = arr2d.astype(np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def _publish_mask_image(self, pub, header, mask_bool: np.ndarray):
        img = Image()
        img.header = header
        img.height = self.V
        img.width = self.H
        img.encoding = "mono8"
        img.is_bigendian = False
        img.step = self.H
        img.data = (mask_bool.astype(np.uint8) * 255).tobytes(order="C")
        pub.publish(img)

    def _mask_nan_outside(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float32, copy=True)
        out[~mask] = np.nan
        return out

    # --------------------
    # PointCloud2 helpers
    # --------------------
    def _pc2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        n = int(msg.width * msg.height)
        if n <= 0:
            return np.empty((0, 3), dtype=np.float32)

        ox = oy = oz = None
        for f in msg.fields:
            if f.name == "x":
                ox = int(f.offset)
            elif f.name == "y":
                oy = int(f.offset)
            elif f.name == "z":
                oz = int(f.offset)

        if ox is None or oy is None or oz is None:
            pts = np.array(
                list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
                dtype=np.float32,
            )
            return pts if pts.size else np.empty((0, 3), dtype=np.float32)

        endian = ">" if msg.is_bigendian else "<"
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
                "offsets": [ox, oy, oz],
                "itemsize": int(msg.point_step),
            }
        )
        arr = np.frombuffer(msg.data, dtype=dtype, count=n)
        pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32, copy=False)
        finite = np.isfinite(pts).all(axis=1)
        return pts[finite]

    def _bin_index(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arctan2(y, x)
        xy = np.sqrt(x * x + y * y)
        phi = np.arctan2(z, xy)

        in_fov = (
            (theta >= self.theta_min) & (theta < self.theta_max) &
            (phi >= self.phi_min) & (phi < self.phi_max) &
            (r > 0.0) & np.isfinite(r)
        )

        h = ((theta - self.theta_min) / self.d_theta).astype(np.int32)
        v = ((phi - self.phi_min) / self.d_phi).astype(np.int32)
        valid = in_fov & (h >= 0) & (h < self.H) & (v >= 0) & (v < self.V)
        return v, h, valid, r

    def _compute_range_and_hits(self, r: np.ndarray, v: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_bins = self.V * self.H
        flat = v * self.H + h
        sum_flat = np.bincount(flat, weights=r, minlength=num_bins).astype(np.float32)
        cnt_flat = np.bincount(flat, minlength=num_bins).astype(np.int32)
        sum2d = sum_flat.reshape(self.V, self.H)
        cnt2d = cnt_flat.reshape(self.V, self.H)

        R = np.zeros((self.V, self.H), dtype=np.float32)
        nz = cnt2d > 0
        R[nz] = sum2d[nz] / cnt2d[nz].astype(np.float32)
        return R, cnt2d.astype(np.float32)

    def _normalize(self, a: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a, dtype=np.float32)
        vals = a[valid_mask]
        if vals.size == 0:
            return out
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        if hi <= lo + 1e-12:
            hi = lo + 1.0
        out[valid_mask] = np.clip((a[valid_mask] - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        return out

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # --------------------
    # Main callback
    # --------------------
    def cb_pc(self, msg: PointCloud2):
        self.prof.frame_start()
        t_cb0 = time.perf_counter_ns()
        self._frame_counter += 1

        # ---- parse points ----
        t0 = time.perf_counter_ns()
        pts = self._pc2_to_xyz(msg)
        t1 = time.perf_counter_ns()
        self.prof.mark("core:convert_pc2_to_xyz", t0, t1)

        if pts.shape[0] == 0:
            proc_time_ms = (time.perf_counter_ns() - t_cb0) / 1e6
            self._csv_write_row(msg.header, n_points=0, proc_time_ms=proc_time_ms)
            self.prof.mark("callback:total", t_cb0, time.perf_counter_ns())
            self.prof.frame_end()
            return

        # ---- binning ----
        t0 = time.perf_counter_ns()
        v, h, valid, r = self._bin_index(pts)
        t1 = time.perf_counter_ns()
        self.prof.mark("core:bin_index", t0, t1)

        if not np.any(valid):
            proc_time_ms = (time.perf_counter_ns() - t_cb0) / 1e6
            self._csv_write_row(msg.header, n_points=int(pts.shape[0]), proc_time_ms=proc_time_ms)
            self.prof.mark("callback:total", t_cb0, time.perf_counter_ns())
            self.prof.frame_end()
            return

        v = v[valid]
        h = h[valid]
        r = r[valid].astype(np.float32, copy=False)

        # ---- range/hit maps ----
        t0 = time.perf_counter_ns()
        R, N = self._compute_range_and_hits(r, v, h)
        M = (N > 0.0)
        t1 = time.perf_counter_ns()
        self.prof.mark("core:range_hits", t0, t1)

        # ---- expected prev ----
        expected_prev = self.expected_range
        hits_prev = self.expected_hits

        min_exp = float(self.get_parameter("min_expected_hits").value)
        enable_temporal = bool(self.get_parameter("enable_temporal").value)

        exp_valid = (hits_prev >= min_exp)
        compare_mask = exp_valid & M
        Omega = exp_valid | M

        # latent R_hat（内部用）
        R_hat = np.where(M, R, expected_prev).astype(np.float32)
        self._last_R_hat = R_hat

        # =========================================================
        # B) Importance head
        # =========================================================
        t0 = time.perf_counter_ns()

        S = np.zeros((self.V, self.H), dtype=np.float32)

        # =========================================================
        # ★修正: 空間特徴量は「観測差分のみ」で算出（期待差分による補完を廃止）
        #       - 旧: both_obs でなければ both_exp のとき expected_prev 差分で補完
        #       - 新: both_obs のときのみ差分を採用し，それ以外は 0
        # =========================================================

        # --- horizontal neighbor diffs (obs-only) ---
        dh_obs = np.abs(R[:, :-1] - R[:, 1:]).astype(np.float32)
        both_obs_h = (M[:, :-1] & M[:, 1:])

        dh = np.zeros_like(dh_obs, dtype=np.float32)
        dh[both_obs_h] = dh_obs[both_obs_h]  # ★修正: 観測されている隣接対のみ

        S[:, :-1] = np.maximum(S[:, :-1], dh)
        S[:, 1:] = np.maximum(S[:, 1:], dh)

        # --- vertical neighbor diffs (obs-only) ---
        dv_obs = np.abs(R[:-1, :] - R[1:, :]).astype(np.float32)
        both_obs_v = (M[:-1, :] & M[1:, :])

        dv = np.zeros_like(dv_obs, dtype=np.float32)
        dv[both_obs_v] = dv_obs[both_obs_v]  # ★修正: 観測されている隣接対のみ

        S[:-1, :] = np.maximum(S[:-1, :], dv)
        S[1:, :] = np.maximum(S[1:, :], dv)

        # --- temporal feature ---
        T = np.zeros((self.V, self.H), dtype=np.float32)
        if enable_temporal:
            T[compare_mask] = np.abs(R[compare_mask] - expected_prev[compare_mask]).astype(np.float32)

        # 正規化（既存どおり）
        S_n = self._normalize(S, Omega)
        T_n = self._normalize(T, compare_mask) if enable_temporal else np.zeros_like(S_n)

        w_s = float(self.get_parameter("w_s").value)
        w_t = float(self.get_parameter("w_t").value)
        I = (w_s * S_n + w_t * T_n).astype(np.float32)

        p = float(self.get_parameter("roi_top_percent").value)
        p = float(np.clip(p, 0.1, 100.0))
        roi_imp = np.zeros_like(M, dtype=bool)
        vals = I[Omega]
        if vals.size > 0:
            thr = np.percentile(vals, 100.0 - p)
            roi_imp = Omega & (I >= thr)

        t1 = time.perf_counter_ns()
        self.prof.mark("core:importance_head", t0, t1)

        # =========================================================
        # C) Reliability head
        # =========================================================
        t0 = time.perf_counter_ns()

        w_m = float(self.get_parameter("w_m").value)
        w_d = float(self.get_parameter("w_d").value)
        w_n = float(self.get_parameter("w_n").value)
        scale_d = float(self.get_parameter("scale_d_m").value)
        n_floor = float(self.get_parameter("n_floor").value)

        d_m = (1.0 - M.astype(np.float32))

        d_d = np.zeros_like(R, dtype=np.float32)
        if scale_d <= 1e-9:
            scale_d = 1.0
        d_d[compare_mask] = np.clip(
            np.abs(R[compare_mask] - expected_prev[compare_mask]) / scale_d, 0.0, 1.0
        ).astype(np.float32)

        denom = np.maximum(hits_prev, n_floor).astype(np.float32)
        d_n = np.clip(np.maximum(hits_prev - N, 0.0) / denom, 0.0, 1.0).astype(np.float32)
        d_n[~exp_valid] = 0.0

        Q = (w_m * d_m + w_d * d_d + w_n * d_n).astype(np.float32)

        beta = float(self.get_parameter("sigmoid_beta").value)
        c = float(self.get_parameter("sigmoid_center_c").value)

        Rel = np.ones_like(Q, dtype=np.float32)
        Rel[Omega] = self._sigmoid(beta * (c - Q[Omega])).astype(np.float32)

        t1 = time.perf_counter_ns()
        self.prof.mark("core:reliability_head", t0, t1)

        # ---- ROI split & scalars ----
        tau_rel = float(self.get_parameter("tau_rel").value)
        roi_use = roi_imp & (Rel >= tau_rel)
        roi_alert = roi_imp & (Rel < tau_rel)

        eps = 1e-6
        alert_ratio = float(roi_alert.sum() / (roi_imp.sum() + eps))
        alert_ratio_omega = float(np.mean((Rel[Omega] < tau_rel), dtype=np.float64)) if np.any(Omega) else float("nan")
        frame_rel_all = float(np.mean(Rel[Omega], dtype=np.float64)) if np.any(Omega) else float("nan")
        frame_rel_obs = float(np.mean(Rel[M], dtype=np.float64)) if np.any(M) else float("nan")

        # ---- stats (Chap7) ----
        omega_bins = int(np.sum(Omega))
        roi_bins = int(np.sum(roi_imp))
        keepbin_ratio = float(roi_bins / max(1, omega_bins))
        n_points = int(pts.shape[0])
        roi_points = int(np.sum(N[roi_imp])) if roi_bins > 0 else 0
        keeppt_ratio = float(roi_points / max(1, n_points))

        if roi_bins > 0:
            s_mean_roi = float(np.mean(S_n[roi_imp], dtype=np.float64))
            t_mean_roi = float(np.mean(T_n[roi_imp], dtype=np.float64))
            i_mean_roi = float(np.mean(I[roi_imp], dtype=np.float64))
        else:
            s_mean_roi = float("nan")
            t_mean_roi = float("nan")
            i_mean_roi = float("nan")

        # ---- publish minimum outputs ----
        t0 = time.perf_counter_ns()

        publish_rel_low = bool(self.get_parameter("publish_rel_low").value)
        if publish_rel_low:
            rel_low = Omega & (Rel < tau_rel)
            self._publish_mask_image(self.pub_rel_low, msg.header, rel_low)

        self._publish_mask_image(self.pub_omega, msg.header, Omega)
        self._publish_mask_image(self.pub_roi_imp, msg.header, roi_imp)
        self._publish_mask_image(self.pub_roi_use, msg.header, roi_use)
        self._publish_mask_image(self.pub_roi_alert, msg.header, roi_alert)

        publish_masked = bool(self.get_parameter("publish_masked_maps").value)
        if publish_masked:
            I_pub = self._mask_nan_outside(I, Omega)
            Rel_pub = self._mask_nan_outside(Rel, Omega)
        else:
            I_pub, Rel_pub = I, Rel

        self._publish_map(self.pub_I, I_pub)
        self._publish_map(self.pub_Rel, Rel_pub)

        m = Float32(); m.data = frame_rel_all; self.pub_frame_rel_all.publish(m)
        m = Float32(); m.data = frame_rel_obs; self.pub_frame_rel_obs.publish(m)
        m = Float32(); m.data = alert_ratio; self.pub_alert_ratio.publish(m)
        m = Float32(); m.data = alert_ratio_omega; self.pub_alert_ratio_omega.publish(m)

        self._stats_write_row(
            msg.header,
            n_points=n_points,
            omega_bins=omega_bins,
            roi_bins=roi_bins,
            keepbin_ratio=keepbin_ratio,
            roi_points=roi_points,
            keeppt_ratio=keeppt_ratio,
            s_mean_roi=s_mean_roi,
            t_mean_roi=t_mean_roi,
            i_mean_roi=i_mean_roi,
            frame_rel_all=frame_rel_all,
            frame_rel_obs=frame_rel_obs,
            alert_ratio=alert_ratio,
            alert_ratio_omega=alert_ratio_omega,
            drop_ratio=self._last_drop_ratio,
            bias_m=self._last_bias_m,
        )

        t1 = time.perf_counter_ns()
        self.prof.mark("core:publish_min", t0, t1)

        # ---- update EMA ----
        t0 = time.perf_counter_ns()
        alpha = float(self.get_parameter("ema_alpha").value)
        hit_mask = (N > 0.0)
        self.expected_hits[hit_mask] = (1.0 - alpha) * self.expected_hits[hit_mask] + alpha * N[hit_mask]
        self.expected_range[hit_mask] = (1.0 - alpha) * self.expected_range[hit_mask] + alpha * R[hit_mask]
        t1 = time.perf_counter_ns()
        self.prof.mark("core:ema_update", t0, t1)

        # ---- CSV: proc_time_ms（callback全体） ----
        proc_time_ms = (time.perf_counter_ns() - t_cb0) / 1e6
        self._csv_write_row(msg.header, n_points=int(pts.shape[0]), proc_time_ms=proc_time_ms)

        self.prof.mark("callback:total", t_cb0, time.perf_counter_ns())
        self.prof.frame_end()

    # --------------------
    # Cleanup
    # --------------------
    def destroy_node(self):
        try:
            if self._csv_fp is not None:
                try:
                    self._csv_fp.flush()
                except Exception:
                    pass
                try:
                    self._csv_fp.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._stats_fp is not None:
                try:
                    self._stats_fp.flush()
                except Exception:
                    pass
                try:
                    self._stats_fp.close()
                except Exception:
                    pass
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImportantROIWithReliability()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RCLError:
            pass


if __name__ == "__main__":
    main()
