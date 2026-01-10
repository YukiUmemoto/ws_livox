#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField, Image  # Image追加
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger  # サービス


# ==============================
# Profiling utilities
# ==============================
@dataclass
class _Stat:
    n: int = 0
    ema_ms: float = 0.0
    max_ms: float = 0.0


class Profiler:
    """
    Lightweight profiler:
      - measures sections in ms
      - keeps EMA and max
      - logs periodically, warns if any section exceeds warn_ms
    """
    def __init__(self, node: Node, enabled: bool, log_period_frames: int, warn_ms: float, ema_alpha: float = 0.1):
        self.node = node
        self.enabled = enabled
        self.log_period_frames = max(1, int(log_period_frames))
        self.warn_ms = float(warn_ms)
        self.ema_alpha = float(ema_alpha)

        self._frame_count = 0
        self._stats = {}       # name -> _Stat
        self._this_frame = {}  # name -> ms

    def frame_start(self):
        if not self.enabled:
            return
        self._this_frame.clear()

    def mark(self, name: str, start_ns: int, end_ns: int):
        if not self.enabled:
            return
        ms = (end_ns - start_ns) / 1e6
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

        # WARN: any section over threshold
        over = [(k, v) for k, v in self._this_frame.items() if v >= self.warn_ms]
        if over:
            over_str = ", ".join([f"{k}={v:.1f}ms" for k, v in sorted(over, key=lambda x: -x[1])])
            self.node.get_logger().warn(f"[PROFILE] slow sections: {over_str}")

        # Periodic summary
        if (self._frame_count % self.log_period_frames) == 0:
            items = sorted(self._stats.items(), key=lambda kv: -kv[1].ema_ms)
            lines = []
            for name, st in items[:60]:
                lines.append(f"{name}: ema={st.ema_ms:.2f}ms, max={st.max_ms:.2f}ms (n={st.n})")
            msg = "\n  ".join(lines)
            self.node.get_logger().info(f"[PROFILE] summary every {self.log_period_frames} frames:\n  {msg}")


class AngleBinEdgeEstimatorEMA(Node):
    def __init__(self):
        super().__init__('angle_bin_edge_estimator')

        # ===== パラメータ =====
        self.declare_parameter('input_topic', '/livox/lidar')

        self.declare_parameter('horizontal_fov_deg', 70.4)
        self.declare_parameter('vertical_fov_deg', 77.2)

        self.declare_parameter('num_horizontal_bins', 128)
        self.declare_parameter('num_vertical_bins', 128)

        self.declare_parameter('temporal_edge_threshold', 0.5)
        self.declare_parameter('spatial_edge_threshold', 0.5)

        self.declare_parameter('temporal_ema_alpha', 0.05)
        self.declare_parameter('min_expected_hits', 3.0)

        self.declare_parameter('visualization_range', 5.0)
        self.declare_parameter('enable_temporal_edge', True)

        # ===== bin_pyramids =====
        self.declare_parameter('enable_bin_pyramids', True)
        self.declare_parameter('bin_pyramids_reset_service', 'edge_estimator_ema/reset_bin_pyramids')

        # ===== Profiling params =====
        self.declare_parameter('profile_enable', True)
        self.declare_parameter('profile_log_period_frames', 30)
        self.declare_parameter('profile_warn_ms', 100.0)
        self.declare_parameter('profile_ema_alpha', 0.1)

        # ===== 可視化(Image mono8) 正規化パラメータ =====
        self.declare_parameter('vis_clip_min', 0.0)
        self.declare_parameter('vis_clip_max', 0.0)
        self.declare_parameter('vis_use_percentile', True)
        self.declare_parameter('vis_p_low', 1.0)
        self.declare_parameter('vis_p_high', 99.0)

        # 入力トピック
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        # パラメータ取得
        self.horizontal_fov_deg = float(self.get_parameter('horizontal_fov_deg').value)
        self.vertical_fov_deg = float(self.get_parameter('vertical_fov_deg').value)
        self.num_horizontal_bins = int(self.get_parameter('num_horizontal_bins').value)
        self.num_vertical_bins = int(self.get_parameter('num_vertical_bins').value)
        self.temporal_edge_threshold = float(self.get_parameter('temporal_edge_threshold').value)
        self.spatial_edge_threshold = float(self.get_parameter('spatial_edge_threshold').value)
        self.temporal_ema_alpha = float(self.get_parameter('temporal_ema_alpha').value)
        self.min_expected_hits = float(self.get_parameter('min_expected_hits').value)
        self.visualization_range = float(self.get_parameter('visualization_range').value)
        self.enable_temporal_edge = bool(self.get_parameter('enable_temporal_edge').value)

        self.enable_bin_pyramids = bool(self.get_parameter('enable_bin_pyramids').value)
        self._bin_pyramids_published = False

        self.bin_pyramids_reset_service = self.get_parameter('bin_pyramids_reset_service').get_parameter_value().string_value

        # 直近frame_id保持（サービスで即時publishするため）
        self._last_frame_id = None

        # Profiling
        self.profile_enable = bool(self.get_parameter('profile_enable').value)
        self.profile_log_period_frames = int(self.get_parameter('profile_log_period_frames').value)
        self.profile_warn_ms = float(self.get_parameter('profile_warn_ms').value)
        self.profile_ema_alpha = float(self.get_parameter('profile_ema_alpha').value)

        self.prof = Profiler(
            node=self,
            enabled=self.profile_enable,
            log_period_frames=self.profile_log_period_frames,
            warn_ms=self.profile_warn_ms,
            ema_alpha=self.profile_ema_alpha
        )

        # 角度範囲 [rad]
        self.theta_min = math.radians(-self.horizontal_fov_deg / 2.0)
        self.theta_max = math.radians(self.horizontal_fov_deg / 2.0)
        self.phi_min = math.radians(-self.vertical_fov_deg / 2.0)
        self.phi_max = math.radians(self.vertical_fov_deg / 2.0)

        self.d_theta = (self.theta_max - self.theta_min) / self.num_horizontal_bins
        self.d_phi = (self.phi_max - self.phi_min) / self.num_vertical_bins

        V = self.num_vertical_bins
        H = self.num_horizontal_bins

        self.expected_map = np.zeros((V, H), dtype=np.float32)
        self.expected_hit_count = np.zeros((V, H), dtype=np.float32)

        self.edge_count_map = np.zeros((V, H), dtype=np.float32)
        self.total_frame_count = 0

        # サブスクライバ
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.pointcloud_callback, 10)

        # パブリッシャ（既存）
        self.header_pub = self.create_publisher(Header, 'edge_estimator_ema/input_header', 10)
        self.range_map_pub = self.create_publisher(Float32MultiArray, 'edge_estimator_ema/range_map', 10)
        self.temporal_edge_map_pub = self.create_publisher(Float32MultiArray, 'edge_estimator_ema/temporal_edge_map', 10)
        self.spatial_edge_map_pub = self.create_publisher(Float32MultiArray, 'edge_estimator_ema/spatial_edge_map', 10)
        self.importance_map_pub = self.create_publisher(Float32MultiArray, 'edge_estimator_ema/importance_map', 10)

        self.edge_cloud_temporal_pub = self.create_publisher(PointCloud2, 'edge_estimator_ema/edge_points_temporal', 10)
        self.edge_cloud_temporal_curr_gt_pub = self.create_publisher(PointCloud2, 'edge_estimator_ema/edge_points_temporal_curr_greater', 10)
        self.edge_cloud_temporal_past_gt_pub = self.create_publisher(PointCloud2, 'edge_estimator_ema/edge_points_temporal_past_greater', 10)

        self.edge_cloud_spatial_pub = self.create_publisher(PointCloud2, 'edge_estimator_ema/edge_points_spatial', 10)
        self.edge_cloud_spatial_front_pub = self.create_publisher(PointCloud2, 'edge_estimator_ema/edge_points_spatial_front', 10)
        self.edge_cloud_spatial_shadow_pub = self.create_publisher(PointCloud2, 'edge_estimator_ema/edge_points_spatial_shadow', 10)

        # ===== 2Dマップ Image 出力 =====
        self.range_img32_pub = self.create_publisher(Image, 'edge_estimator_ema/range_map_img32', 10)
        self.range_mono8_pub = self.create_publisher(Image, 'edge_estimator_ema/range_map_mono8', 10)

        self.spatial_img32_pub = self.create_publisher(Image, 'edge_estimator_ema/spatial_edge_map_img32', 10)
        self.spatial_mono8_pub = self.create_publisher(Image, 'edge_estimator_ema/spatial_edge_map_mono8', 10)

        self.temporal_img32_pub = self.create_publisher(Image, 'edge_estimator_ema/temporal_edge_map_img32', 10)
        self.temporal_mono8_pub = self.create_publisher(Image, 'edge_estimator_ema/temporal_edge_map_mono8', 10)

        self.importance_img32_pub = self.create_publisher(Image, 'edge_estimator_ema/importance_map_img32', 10)
        self.importance_mono8_pub = self.create_publisher(Image, 'edge_estimator_ema/importance_map_mono8', 10)

        # bin_pyramids
        if self.enable_bin_pyramids:
            self.bin_pyramids_pub = self.create_publisher(Marker, 'edge_estimator_ema/bin_pyramids', 10)
            self.bin_pyramid_points = self.generate_bin_pyramid_edges_points()  # 生成は1回だけ
        else:
            self.bin_pyramids_pub = None
            self.bin_pyramid_points = None

        # resetサービス
        self.reset_srv = self.create_service(Trigger, self.bin_pyramids_reset_service, self._on_reset_bin_pyramids)

        self.get_logger().info(
            f'AngleBinEdgeEstimatorEMA started. '
            f'Subscribe: {self.input_topic}, '
            f'alpha={self.temporal_ema_alpha}, '
            f'min_expected_hits={self.min_expected_hits}, '
            f'enable_temporal_edge={self.enable_temporal_edge}, '
            f'enable_bin_pyramids={self.enable_bin_pyramids}, '
            f'profile_enable={self.profile_enable}, '
            f'warn_ms={self.profile_warn_ms}, '
            f'log_period_frames={self.profile_log_period_frames}, '
            f'reset_service={self.bin_pyramids_reset_service}'
        )

    # =========================
    # map(2D float32) → Image(32FC1, mono8) publish
    # =========================
    def _publish_map_images(self, arr2d: np.ndarray, header_in: Header,
                            pub32: Image, pub8: Image, name: str):
        if arr2d is None:
            return
        a = np.asarray(arr2d, dtype=np.float32)
        if a.ndim != 2:
            self.get_logger().warn(f"{name}: not 2D array: shape={a.shape}")
            return

        V, H = a.shape

        # ---- 32FC1（評価用：値そのまま） ----
        msg32 = Image()
        msg32.header = header_in
        msg32.height = V
        msg32.width = H
        msg32.encoding = '32FC1'
        msg32.is_bigendian = False
        msg32.step = H * 4
        msg32.data = a.tobytes(order='C')
        pub32.publish(msg32)

        # ---- mono8（可視化用：0-255） ----
        use_percentile = bool(self.get_parameter('vis_use_percentile').value)
        clip_min = float(self.get_parameter('vis_clip_min').value)
        clip_max = float(self.get_parameter('vis_clip_max').value)
        p_low = float(self.get_parameter('vis_p_low').value)
        p_high = float(self.get_parameter('vis_p_high').value)

        a_vis = a.copy()
        a_vis[~np.isfinite(a_vis)] = 0.0

        # 可視化用だけ上下・左右反転（180°回転）
        a_vis = np.flipud(a_vis)
        a_vis = np.fliplr(a_vis)

        if clip_min == 0.0 and clip_max == 0.0:
            if use_percentile:
                nonzero = a_vis[np.abs(a_vis) > 0]
                if nonzero.size > 0:
                    lo = float(np.percentile(nonzero, p_low))
                    hi = float(np.percentile(nonzero, p_high))
                else:
                    lo, hi = 0.0, 1.0
            else:
                lo, hi = float(np.min(a_vis)), float(np.max(a_vis))
        else:
            lo, hi = clip_min, clip_max

        if hi <= lo + 1e-12:
            hi = lo + 1.0

        a_vis = np.clip(a_vis, lo, hi)
        a_vis = (a_vis - lo) / (hi - lo)
        a_u8 = (a_vis * 255.0).astype(np.uint8)

        msg8 = Image()
        msg8.header = header_in
        msg8.height = V
        msg8.width = H
        msg8.encoding = 'mono8'
        msg8.is_bigendian = False
        msg8.step = H
        msg8.data = a_u8.tobytes(order='C')
        pub8.publish(msg8)

    # =========================
    # resetサービス
    # =========================
    def _on_reset_bin_pyramids(self, request: Trigger.Request, response: Trigger.Response):
        if not self.enable_bin_pyramids or self.bin_pyramids_pub is None or self.bin_pyramid_points is None:
            response.success = False
            response.message = "bin_pyramids is disabled or not initialized."
            return response

        self._bin_pyramids_published = False

        if self._last_frame_id is not None and self._last_frame_id != "":
            header = Header()
            header.frame_id = self._last_frame_id
            header.stamp = self.get_clock().now().to_msg()
            self.publish_bin_pyramids(header)
            self._bin_pyramids_published = True
            response.success = True
            response.message = f"bin_pyramids republished once (frame_id={self._last_frame_id})."
        else:
            response.success = True
            response.message = "reset done. frame_id unknown; will publish once on next incoming frame."
        return response

    # ========== メインコールバック ==========
    def pointcloud_callback(self, msg: PointCloud2):
        self.prof.frame_start()
        t_cb0 = time.perf_counter_ns()

        if msg.header.frame_id:
            self._last_frame_id = msg.header.frame_id

        # 入力ヘッダ publish
        t0 = time.perf_counter_ns()
        self.header_pub.publish(msg.header)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:input_header", t0, t1)

        # PointCloud2 → numpy array (x, y, z)
        t0 = time.perf_counter_ns()
        points = self.pointcloud2_to_xyz_array(msg)
        t1 = time.perf_counter_ns()
        self.prof.mark("convert:pc2_to_xyz", t0, t1)

        if points.shape[0] == 0:
            self._publish_bin_pyramids_once(msg.header)
            self._profile_callback_total(t_cb0)
            return

        # (0,0,0) 除外
        t0 = time.perf_counter_ns()
        non_zero_mask = ~np.all(points == 0.0, axis=1)
        points = points[non_zero_mask]
        t1 = time.perf_counter_ns()
        self.prof.mark("filter:non_zero", t0, t1)

        if points.shape[0] == 0:
            self._publish_bin_pyramids_once(msg.header)
            self._profile_callback_total(t_cb0)
            return

        # 球座標
        t0 = time.perf_counter_ns()
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        xy_norm = np.sqrt(x**2 + y**2)
        phi = np.arctan2(z, xy_norm)
        t1 = time.perf_counter_ns()
        self.prof.mark("prep:spherical", t0, t1)

        # FOV
        t0 = time.perf_counter_ns()
        mask_fov = (
            (theta >= self.theta_min) & (theta < self.theta_max) &
            (phi >= self.phi_min) & (phi < self.phi_max)
        )
        t1 = time.perf_counter_ns()
        self.prof.mark("filter:fov", t0, t1)

        if not np.any(mask_fov):
            self._publish_bin_pyramids_once(msg.header)
            self._profile_callback_total(t_cb0)
            return

        r = r[mask_fov]
        theta = theta[mask_fov]
        phi = phi[mask_fov]
        points_fov = points[mask_fov]

        # ビンindex
        t0 = time.perf_counter_ns()
        h_idx = ((theta - self.theta_min) / self.d_theta).astype(np.int32)
        v_idx = ((phi - self.phi_min) / self.d_phi).astype(np.int32)

        valid_mask = (
            (h_idx >= 0) & (h_idx < self.num_horizontal_bins) &
            (v_idx >= 0) & (v_idx < self.num_vertical_bins)
        )
        t1 = time.perf_counter_ns()
        self.prof.mark("prep:bin_index+valid", t0, t1)

        if not np.any(valid_mask):
            self._publish_bin_pyramids_once(msg.header)
            self._profile_callback_total(t_cb0)
            return

        h_idx = h_idx[valid_mask]
        v_idx = v_idx[valid_mask]
        r = r[valid_mask]
        points_fov = points_fov[valid_mask]

        # レンジマップ
        t0 = time.perf_counter_ns()
        range_map, hit_count_map = self.compute_range_map(r, v_idx, h_idx)
        t1 = time.perf_counter_ns()
        self.prof.mark("compute:range_map", t0, t1)

        t0 = time.perf_counter_ns()
        self.publish_range_map(self.range_map_pub, range_map)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:range_map", t0, t1)

        # range_map を Image publish
        t0 = time.perf_counter_ns()
        self._publish_map_images(range_map, msg.header,
                                 self.range_img32_pub, self.range_mono8_pub,
                                 name="range_map")
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:range_map_images", t0, t1)

        # ============================================================
        # ★時間軸比較に変更：
        #   - EMA更新"前"の expected_prev を「過去」とみなして比較する
        #   - 比較 → split → publish → 最後に EMA更新
        # ============================================================
        expected_prev = self.expected_map.copy()
        hit_prev = self.expected_hit_count.copy()

        temporal_edge_map = np.zeros_like(range_map, dtype=np.float32)
        temporal_edge_mask = None

        if self.enable_temporal_edge:
            t0 = time.perf_counter_ns()

            valid_expectation = hit_prev >= self.min_expected_hits
            current_valid = hit_count_map > 0
            compare_mask = valid_expectation & current_valid

            # 過去(expected_prev)との差分
            delta_t = range_map - expected_prev
            temporal_edge_map[compare_mask] = np.abs(delta_t[compare_mask])

            temporal_edge_mask = np.zeros_like(range_map, dtype=bool)
            temporal_edge_mask[compare_mask] = (temporal_edge_map[compare_mask] > self.temporal_edge_threshold)

            t1 = time.perf_counter_ns()
            self.prof.mark("compute:temporal_edge(time_compare)", t0, t1)

            # temporal_edge_map publish
            t0 = time.perf_counter_ns()
            self.publish_range_map(self.temporal_edge_map_pub, temporal_edge_map)
            t1 = time.perf_counter_ns()
            self.prof.mark("pub:temporal_edge_map", t0, t1)

            # temporal_edge_map Image publish
            t0 = time.perf_counter_ns()
            self._publish_map_images(temporal_edge_map, msg.header,
                                     self.temporal_img32_pub, self.temporal_mono8_pub,
                                     name="temporal_edge_map")
            t1 = time.perf_counter_ns()
            self.prof.mark("pub:temporal_edge_map_images", t0, t1)

            # edge points（全部）
            t0 = time.perf_counter_ns()
            self.publish_edge_points(points_fov, v_idx, h_idx, temporal_edge_mask, msg.header, self.edge_cloud_temporal_pub)
            t1 = time.perf_counter_ns()
            self.prof.mark("pub:edge_points_temporal", t0, t1)

            # ------------------------------------------------------------
            # ★split（トピック名は変更しない）
            #   直感に合わせて：
            #     - 人（現在が近い：delta_t < 0） → curr_greater 側へ入れる
            #     - 背景（現在が遠い：delta_t > 0） → past_greater 側へ入れる
            #
            #   ※トピック名はそのままですが、中身の割当を入れ替えています。
            # ------------------------------------------------------------
            t0 = time.perf_counter_ns()
            temporal_curr_gt_bins = np.zeros_like(temporal_edge_mask, dtype=bool)
            temporal_past_gt_bins = np.zeros_like(temporal_edge_mask, dtype=bool)

            # 現在が近い（人が出現しやすい） → curr_greater（赤にしたい方）
            temporal_curr_gt_bins[temporal_edge_mask & (delta_t < 0.0)] = True
            # 現在が遠い（背景が出る等） → past_greater（青にしたい方）
            temporal_past_gt_bins[temporal_edge_mask & (delta_t > 0.0)] = True
            t1 = time.perf_counter_ns()
            self.prof.mark("compute:temporal_split(reassigned)", t0, t1)

            t0 = time.perf_counter_ns()
            self.publish_edge_points(points_fov, v_idx, h_idx, temporal_curr_gt_bins, msg.header, self.edge_cloud_temporal_curr_gt_pub)
            t1 = time.perf_counter_ns()
            self.prof.mark("pub:edge_points_temporal_curr_gt", t0, t1)

            t0 = time.perf_counter_ns()
            self.publish_edge_points(points_fov, v_idx, h_idx, temporal_past_gt_bins, msg.header, self.edge_cloud_temporal_past_gt_pub)
            t1 = time.perf_counter_ns()
            self.prof.mark("pub:edge_points_temporal_past_gt", t0, t1)

        # ---- EMA更新（最後に行う：次フレーム以降の“過去”になる） ----
        t0 = time.perf_counter_ns()
        alpha = self.temporal_ema_alpha
        hit_mask = hit_count_map > 0
        self.expected_hit_count[hit_mask] = (1.0 - alpha) * self.expected_hit_count[hit_mask] + alpha * hit_count_map[hit_mask]
        self.expected_map[hit_mask] = (1.0 - alpha) * self.expected_map[hit_mask] + alpha * range_map[hit_mask]
        t1 = time.perf_counter_ns()
        self.prof.mark("compute:ema_update(after_time_compare)", t0, t1)

        # spatial edge
        t0 = time.perf_counter_ns()
        spatial_edge_map, spatial_edge_mask, front_bins, shadow_bins = self.compute_spatial_edge(range_map)
        t1 = time.perf_counter_ns()
        self.prof.mark("compute:spatial_edge", t0, t1)

        t0 = time.perf_counter_ns()
        self.publish_range_map(self.spatial_edge_map_pub, spatial_edge_map)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:spatial_edge_map", t0, t1)

        # spatial_edge_map Image publish
        t0 = time.perf_counter_ns()
        self._publish_map_images(spatial_edge_map, msg.header,
                                 self.spatial_img32_pub, self.spatial_mono8_pub,
                                 name="spatial_edge_map")
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:spatial_edge_map_images", t0, t1)

        t0 = time.perf_counter_ns()
        self.publish_edge_points(points_fov, v_idx, h_idx, spatial_edge_mask, msg.header, self.edge_cloud_spatial_pub)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:edge_points_spatial", t0, t1)

        t0 = time.perf_counter_ns()
        self.publish_edge_points(points_fov, v_idx, h_idx, front_bins, msg.header, self.edge_cloud_spatial_front_pub)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:edge_points_spatial_front", t0, t1)

        t0 = time.perf_counter_ns()
        self.publish_edge_points(points_fov, v_idx, h_idx, shadow_bins, msg.header, self.edge_cloud_spatial_shadow_pub)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:edge_points_spatial_shadow", t0, t1)

        # importance
        t0 = time.perf_counter_ns()
        self.total_frame_count += 1
        self.edge_count_map[spatial_edge_mask] += 1.0
        importance_map = self.edge_count_map / max(self.total_frame_count, 1)
        t1 = time.perf_counter_ns()
        self.prof.mark("compute:importance", t0, t1)

        t0 = time.perf_counter_ns()
        self.publish_range_map(self.importance_map_pub, importance_map)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:importance_map", t0, t1)

        # importance_map Image publish
        t0 = time.perf_counter_ns()
        self._publish_map_images(importance_map, msg.header,
                                 self.importance_img32_pub, self.importance_mono8_pub,
                                 name="importance_map")
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:importance_map_images", t0, t1)

        # bin_pyramids（1回だけ）
        self._publish_bin_pyramids_once(msg.header)

        self._profile_callback_total(t_cb0)

    def _publish_bin_pyramids_once(self, header_in: Header):
        if not self.enable_bin_pyramids:
            return
        if self._bin_pyramids_published:
            return
        if self.bin_pyramids_pub is None or self.bin_pyramid_points is None:
            return

        t0 = time.perf_counter_ns()
        self.publish_bin_pyramids(header_in)
        t1 = time.perf_counter_ns()
        self.prof.mark("pub:bin_pyramids_once", t0, t1)

        self._bin_pyramids_published = True

    def _profile_callback_total(self, t_cb0: int):
        t_cb1 = time.perf_counter_ns()
        self.prof.mark("callback:total", t_cb0, t_cb1)
        self.prof.frame_end()

    # ========== レンジマップ ==========
    def compute_range_map(self, r: np.ndarray, v_idx: np.ndarray, h_idx: np.ndarray):
        V = self.num_vertical_bins
        H = self.num_horizontal_bins
        num_bins = V * H

        flat_idx = v_idx * H + h_idx

        range_sum_flat = np.bincount(flat_idx, weights=r, minlength=num_bins).astype(np.float32)
        range_count_flat = np.bincount(flat_idx, minlength=num_bins).astype(np.int32)

        range_sum = range_sum_flat.reshape(V, H)
        range_count = range_count_flat.reshape(V, H)

        range_map = np.zeros((V, H), dtype=np.float32)
        non_empty = range_count > 0
        range_map[non_empty] = range_sum[non_empty] / range_count[non_empty]
        return range_map, range_count

    # ========== 空間エッジ ==========
    def compute_spatial_edge(self, range_map: np.ndarray):
        diff_right = np.zeros_like(range_map)
        diff_left = np.zeros_like(range_map)
        diff_down = np.zeros_like(range_map)
        diff_up = np.zeros_like(range_map)

        diff_right[:, :-1] = np.abs(range_map[:, :-1] - range_map[:, 1:])
        diff_left[:, 1:] = np.abs(range_map[:, 1:] - range_map[:, :-1])
        diff_down[:-1, :] = np.abs(range_map[:-1, :] - range_map[1:, :])
        diff_up[1:, :] = np.abs(range_map[1:, :] - range_map[:-1, :])

        spatial_edge_map = np.maximum.reduce([diff_right, diff_left, diff_down, diff_up])

        no_obs = (range_map == 0.0)
        spatial_edge_map[no_obs] = 0.0

        spatial_edge_mask = spatial_edge_map > self.spatial_edge_threshold

        front_bins = np.zeros_like(spatial_edge_mask, dtype=bool)
        shadow_bins = np.zeros_like(spatial_edge_mask, dtype=bool)

        both_valid_h = (range_map[:, :-1] > 0.0) & (range_map[:, 1:] > 0.0)
        diff_pair_h = np.abs(range_map[:, :-1] - range_map[:, 1:])
        cond_edge_h = both_valid_h & (diff_pair_h > self.spatial_edge_threshold)

        left_near = cond_edge_h & (range_map[:, :-1] < range_map[:, 1:])
        right_near = cond_edge_h & (range_map[:, 1:] < range_map[:, :-1])

        front_bins[:, :-1][left_near] = True
        shadow_bins[:, 1:][left_near] = True
        front_bins[:, 1:][right_near] = True
        shadow_bins[:, :-1][right_near] = True

        both_valid_v = (range_map[:-1, :] > 0.0) & (range_map[1:, :] > 0.0)
        diff_pair_v = np.abs(range_map[:-1, :] - range_map[1:, :])
        cond_edge_v = both_valid_v & (diff_pair_v > self.spatial_edge_threshold)

        up_near = cond_edge_v & (range_map[:-1, :] < range_map[1:, :])
        down_near = cond_edge_v & (range_map[1:, :] < range_map[:-1, :])

        front_bins[:-1, :][up_near] = True
        shadow_bins[1:, :][up_near] = True
        front_bins[1:, :][down_near] = True
        shadow_bins[:-1, :][down_near] = True

        front_bins &= spatial_edge_mask
        shadow_bins &= spatial_edge_mask

        return spatial_edge_map, spatial_edge_mask, front_bins, shadow_bins

    # ========== 2Dマップ publish ==========
    def publish_range_map(self, pub, arr: np.ndarray):
        msg = Float32MultiArray()

        dim_v = MultiArrayDimension()
        dim_v.label = 'vertical'
        dim_v.size = self.num_vertical_bins
        dim_v.stride = self.num_vertical_bins * self.num_horizontal_bins

        dim_h = MultiArrayDimension()
        dim_h.label = 'horizontal'
        dim_h.size = self.num_horizontal_bins
        dim_h.stride = self.num_horizontal_bins

        msg.layout.dim = [dim_v, dim_h]
        msg.layout.data_offset = 0
        msg.data = arr.flatten().tolist()
        pub.publish(msg)

    # ========== エッジ点群 publish ==========
    def publish_edge_points(self, points_fov, v_idx, h_idx, edge_mask_bins, header_in, pub):
        if edge_mask_bins is None or not np.any(edge_mask_bins):
            return

        H = self.num_horizontal_bins
        flat_idx = v_idx * H + h_idx
        edge_mask_flat = edge_mask_bins.reshape(-1)
        point_edge_mask = edge_mask_flat[flat_idx]
        if not np.any(point_edge_mask):
            return

        edge_points = points_fov[point_edge_mask]
        edge_msg = self.xyz_array_to_pointcloud2(edge_points, frame_id=header_in.frame_id)
        edge_msg.header.stamp = header_in.stamp
        pub.publish(edge_msg)

    # ========== bin_pyramids の線分生成 ==========
    def generate_bin_pyramid_edges_points(self):
        points = []
        r = self.visualization_range

        origin = Point()
        origin.x = 0.0
        origin.y = 0.0
        origin.z = 0.0

        def sph_to_cart(radius, theta, phi):
            p = Point()
            p.x = radius * math.cos(phi) * math.cos(theta)
            p.y = radius * math.cos(phi) * math.sin(theta)
            p.z = radius * math.sin(phi)
            return p

        for i in range(self.num_horizontal_bins):
            theta0 = self.theta_min + i * self.d_theta
            theta1 = theta0 + self.d_theta

            for j in range(self.num_vertical_bins):
                phi0 = self.phi_min + j * self.d_phi
                phi1 = phi0 + self.d_phi

                c1 = sph_to_cart(r, theta0, phi0)
                c2 = sph_to_cart(r, theta1, phi0)
                c3 = sph_to_cart(r, theta1, phi1)
                c4 = sph_to_cart(r, theta0, phi1)

                points.append(origin); points.append(c1)
                points.append(origin); points.append(c2)
                points.append(origin); points.append(c3)
                points.append(origin); points.append(c4)

                points.append(c1); points.append(c2)
                points.append(c2); points.append(c3)
                points.append(c3); points.append(c4)
                points.append(c4); points.append(c1)

        return points

    # ========== bin_pyramids publish ==========
    def publish_bin_pyramids(self, header_in: Header):
        marker = Marker()
        marker.header.frame_id = header_in.frame_id
        marker.header.stamp = header_in.stamp
        marker.ns = "angle_bin_pyramids_ema"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.005

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.points = self.bin_pyramid_points
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        self.bin_pyramids_pub.publish(marker)

    # ==============================
    # PointCloud2 -> numpy 高速化
    # ==============================
    def pointcloud2_to_xyz_array(self, cloud_msg: PointCloud2) -> np.ndarray:
        try:
            if cloud_msg.point_step < 12:
                raise ValueError("point_step < 12")

            n = cloud_msg.width * cloud_msg.height
            if n == 0:
                return np.empty((0, 3), dtype=np.float32)

            endian = ">" if cloud_msg.is_bigendian else "<"
            xyz_dtype = np.dtype([("x", f"{endian}f4"), ("y", f"{endian}f4"), ("z", f"{endian}f4")])

            buf = np.frombuffer(cloud_msg.data, dtype=np.uint8)
            expected = n * cloud_msg.point_step
            if buf.size < expected:
                raise ValueError(f"data size mismatch: {buf.size} < {expected}")

            buf2 = buf[:expected].reshape(n, cloud_msg.point_step)
            xyz_bytes = buf2[:, :12].copy()
            xyz = xyz_bytes.view(xyz_dtype).reshape(n)
            points = np.stack([xyz["x"], xyz["y"], xyz["z"]], axis=1).astype(np.float32, copy=False)

            finite = np.isfinite(points).all(axis=1)
            if not np.all(finite):
                points = points[finite]
            return points

        except Exception:
            pts = []
            for p in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append([p[0], p[1], p[2]])
            if len(pts) == 0:
                return np.empty((0, 3), dtype=np.float32)
            return np.array(pts, dtype=np.float32)

    # ========== numpy -> PointCloud2 ==========
    def xyz_array_to_pointcloud2(self, xyz: np.ndarray, frame_id: str) -> PointCloud2:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        return point_cloud2.create_cloud(header, fields, xyz.tolist())


def main(args=None):
    rclpy.init(args=args)
    node = AngleBinEdgeEstimatorEMA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

