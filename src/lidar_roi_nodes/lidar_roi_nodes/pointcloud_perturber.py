#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float32
from sensor_msgs_py import point_cloud2

from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError


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


def _field_offset(fields, name: str) -> Optional[int]:
    for f in fields:
        if f.name == name:
            return int(f.offset)
    return None


def pc2_buffer_as_xyz_view(msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    PointCloud2 のバイト列を「点単位の構造体」として view し、x/y/z を抽出する。
    返り値: (arr_struct, x, y, z, endian)
      - arr_struct: itemsize=point_step の構造体配列（コピーなし）
      - x/y/z: コピーなし view（ただし後で書き換える場合は copy が必要）
    """
    n = int(msg.width * msg.height)
    if n <= 0:
        raise ValueError("Empty PointCloud2")

    ox = _field_offset(msg.fields, "x")
    oy = _field_offset(msg.fields, "y")
    oz = _field_offset(msg.fields, "z")
    if ox is None or oy is None or oz is None:
        raise ValueError("PointCloud2 must have x,y,z fields")

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
    return arr, arr["x"], arr["y"], arr["z"], endian


class PointCloudPerturber(Node):
    def __init__(self):
        super().__init__("pointcloud_perturber")

        # ======================
        # QoS params
        # ======================
        self.declare_parameter("sub_reliability", "best_effort")
        self.declare_parameter("sub_durability", "volatile")
        self.declare_parameter("sub_history", "keep_last")
        self.declare_parameter("sub_depth", 10)

        self.declare_parameter("pub_reliability", "best_effort")
        self.declare_parameter("pub_durability", "volatile")
        self.declare_parameter("pub_history", "keep_last")
        self.declare_parameter("pub_depth", 10)

        # ======================
        # Topics
        # ======================
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("output_topic", "/livox/lidar_perturbed")

        self.declare_parameter("gt_mask_topic", "pc_perturber/gt_mask_mono8")
        self.declare_parameter("drop_ratio_topic", "pc_perturber/drop_ratio")
        self.declare_parameter("bias_topic", "pc_perturber/bias_m")

        # ======================
        # Binning params (GT mask)
        # ======================
        self.declare_parameter("horizontal_fov_deg", 70.4)
        self.declare_parameter("vertical_fov_deg", 77.2)
        self.declare_parameter("vertical_fov_up_deg", -1.0)
        self.declare_parameter("vertical_fov_down_deg", -1.0)
        self.declare_parameter("azimuth_0_to_hfov", True)
        self.declare_parameter("num_horizontal_bins", 128)
        self.declare_parameter("num_vertical_bins", 128)

        # ======================
        # Drop params
        # ======================
        self.declare_parameter("enable_drop", True)
        self.declare_parameter("packet_points", 100)     # Livox packet 100 pts (approx)
        self.declare_parameter("drop_mode", "random")    # random / burst
        self.declare_parameter("drop_prob_q", 0.0)       # random drop prob
        self.declare_parameter("burst_len_B", 5)         # burst length in packets
        self.declare_parameter("burst_start_prob", 0.02) # burst start probability per packet
        self.declare_parameter("rng_seed", 0)

        # ======================
        # Spoof params
        # ======================
        self.declare_parameter("enable_spoof", False)
        self.declare_parameter("spoof_bias_m", 0.5)  # Δ [m]
        # spoof region specified by bin rectangle
        self.declare_parameter("spoof_h_min", 40)
        self.declare_parameter("spoof_h_max", 60)  # exclusive
        self.declare_parameter("spoof_v_min", 50)
        self.declare_parameter("spoof_v_max", 80)  # exclusive

        # ======================
        # Setup
        # ======================
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.Hfov = float(self.get_parameter("horizontal_fov_deg").value)
        self.Vfov = float(self.get_parameter("vertical_fov_deg").value)
        self.Vfov_up = float(self.get_parameter("vertical_fov_up_deg").value)
        self.Vfov_down = float(self.get_parameter("vertical_fov_down_deg").value)
        self.azimuth_0_to_hfov = bool(self.get_parameter("azimuth_0_to_hfov").value)
        self.H = int(self.get_parameter("num_horizontal_bins").value)
        self.V = int(self.get_parameter("num_vertical_bins").value)

        self.theta_min = math.radians(0.0 if self.azimuth_0_to_hfov else (-self.Hfov / 2.0))
        self.theta_max = math.radians(self.Hfov if self.azimuth_0_to_hfov else (+self.Hfov / 2.0))
        if self.Vfov_up > 0.0 and self.Vfov_down > 0.0 and (self.Vfov_up + self.Vfov_down) > 0.0:
            self.phi_min = math.radians(-self.Vfov_down)
            self.phi_max = math.radians(+self.Vfov_up)
            self.Vfov = self.Vfov_up + self.Vfov_down
        else:
            self.phi_min = math.radians(-self.Vfov / 2.0)
            self.phi_max = math.radians(+self.Vfov / 2.0)
        self.d_theta = (self.theta_max - self.theta_min) / self.H
        self.d_phi = (self.phi_max - self.phi_min) / self.V

        seed = int(self.get_parameter("rng_seed").value)
        self.rng = np.random.default_rng(seed if seed != 0 else None)

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

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cb, sub_qos)
        self.pub_pc = self.create_publisher(PointCloud2, self.output_topic, pub_qos)

        self.pub_gt = self.create_publisher(Image, self.get_parameter("gt_mask_topic").value, pub_qos)
        self.pub_drop = self.create_publisher(Float32, self.get_parameter("drop_ratio_topic").value, pub_qos)
        self.pub_bias = self.create_publisher(Float32, self.get_parameter("bias_topic").value, pub_qos)

        self.get_logger().info(
            f"PointCloudPerturber started.\n"
            f"  input={self.input_topic}\n"
            f"  output={self.output_topic}\n"
            f"  bins=VxH={self.V}x{self.H}\n"
        )

    def _bin_index(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.sqrt(x * x + y * y + z * z)
        if self.azimuth_0_to_hfov:
            theta = np.mod(np.arctan2(y, x), 2.0 * math.pi)
        else:
            theta = np.arctan2(y, x)
        xy = np.sqrt(x * x + y * y)
        phi = np.arctan2(z, xy)

        in_fov = (
            (theta >= self.theta_min) & (theta < self.theta_max) &
            (phi >= self.phi_min) & (phi < self.phi_max) &
            np.isfinite(r) & (r > 0.0)
        )

        h = ((theta - self.theta_min) / self.d_theta).astype(np.int32)
        v = ((phi - self.phi_min) / self.d_phi).astype(np.int32)

        valid = in_fov & (h >= 0) & (h < self.H) & (v >= 0) & (v < self.V)
        return v, h, valid

    def _publish_gt_mask(self, header, gt_mask_bool: np.ndarray):
        # mono8 image: 0/255
        img = Image()
        img.header = header
        img.height = int(self.V)
        img.width = int(self.H)
        img.encoding = "mono8"
        img.is_bigendian = False
        img.step = int(self.H)
        data = (gt_mask_bool.astype(np.uint8) * 255)
        # 可視化向けに 180度回転（上下左右反転）する場合はここで flip してください
        img.data = data.tobytes(order="C")
        self.pub_gt.publish(img)

    def cb(self, msg: PointCloud2):
        t0 = time.perf_counter_ns()

        enable_drop = bool(self.get_parameter("enable_drop").value)
        enable_spoof = bool(self.get_parameter("enable_spoof").value)

        packet_points = int(self.get_parameter("packet_points").value)
        drop_mode = str(self.get_parameter("drop_mode").value).lower()
        q = float(self.get_parameter("drop_prob_q").value)
        B = int(self.get_parameter("burst_len_B").value)
        p_start = float(self.get_parameter("burst_start_prob").value)

        bias_m = float(self.get_parameter("spoof_bias_m").value)
        hmin = int(self.get_parameter("spoof_h_min").value)
        hmax = int(self.get_parameter("spoof_h_max").value)
        vmin = int(self.get_parameter("spoof_v_min").value)
        vmax = int(self.get_parameter("spoof_v_max").value)

        # ---- parse point cloud buffer (x,y,z view) ----
        n = int(msg.width * msg.height)
        if n <= 0:
            return

        try:
            arr, x, y, z, _endian = pc2_buffer_as_xyz_view(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse xyz from PointCloud2: {e}")
            return

        # ---- build GT mask (bool VxH) ----
        gt_mask = np.zeros((self.V, self.H), dtype=bool)

        # ---- compute bins for all points (for spoof & drop GT bookkeeping) ----
        v_idx, h_idx, valid = self._bin_index(x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))

        # ---- decide drop mask in "packet-like" 100-pt chunks ----
        keep = np.ones(n, dtype=bool)
        if enable_drop and q > 0.0:
            m = int(math.ceil(n / max(1, packet_points)))  # packets count
            pkt_keep = np.ones(m, dtype=bool)

            if drop_mode == "random":
                drop_pkt = self.rng.random(m) < q
                pkt_keep[drop_pkt] = False

            elif drop_mode == "burst":
                i = 0
                while i < m:
                    if self.rng.random() < p_start:
                        j = min(m, i + max(1, B))
                        pkt_keep[i:j] = False
                        i = j
                    else:
                        i += 1
            else:
                self.get_logger().warn(f"Unknown drop_mode={drop_mode}; fallback random")
                drop_pkt = self.rng.random(m) < q
                pkt_keep[drop_pkt] = False

            # expand to point-level keep
            for pi in range(m):
                s = pi * packet_points
                e = min(n, (pi + 1) * packet_points)
                keep[s:e] = pkt_keep[pi]

            # GT: dropped points' bins are "modified"
            dropped = ~keep
            if np.any(dropped & valid):
                gt_mask[v_idx[dropped & valid], h_idx[dropped & valid]] = True

        # ---- spoof (range bias) in a bin-rectangle region ----
        if enable_spoof and abs(bias_m) > 1e-9:
            in_rect = valid & (h_idx >= hmin) & (h_idx < hmax) & (v_idx >= vmin) & (v_idx < vmax)
            if np.any(in_rect):
                # mark GT bins
                gt_mask[v_idx[in_rect], h_idx[in_rect]] = True

                # apply bias in-place: scale xyz by (r+Δ)/r
                # NOTE: keep_mask applied later; we bias before dropping rows (either ok)
                xx = x[in_rect].astype(np.float32)
                yy = y[in_rect].astype(np.float32)
                zz = z[in_rect].astype(np.float32)
                rr = np.sqrt(xx * xx + yy * yy + zz * zz)
                rr = np.maximum(rr, 1e-6)
                scale = (rr + bias_m) / rr
                # to write back, we need a writable copy of buffer.
                # easiest: create a writable bytearray, then update x/y/z via structured view
                # Here we do: copy message data to mutable, then view and modify.
                buf = np.frombuffer(msg.data, dtype=np.uint8).copy()
                # rebuild struct view on mutable buffer
                ox = _field_offset(msg.fields, "x")
                oy = _field_offset(msg.fields, "y")
                oz = _field_offset(msg.fields, "z")
                endian = ">" if msg.is_bigendian else "<"
                dtype_full = np.dtype(
                    {
                        "names": ["x", "y", "z"],
                        "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
                        "offsets": [ox, oy, oz],
                        "itemsize": int(msg.point_step),
                    }
                )
                arr_mut = np.frombuffer(buf, dtype=dtype_full, count=n)
                arr_mut["x"][in_rect] = (xx * scale).astype(np.float32)
                arr_mut["y"][in_rect] = (yy * scale).astype(np.float32)
                arr_mut["z"][in_rect] = (zz * scale).astype(np.float32)

                # replace msg.data with mutated (temporary) for subsequent dropping
                data_mut = buf.tobytes()
            else:
                data_mut = msg.data
        else:
            data_mut = msg.data

        # ---- apply dropping (row filtering) while keeping all original fields ----
        if np.any(~keep):
            buf_all = np.frombuffer(data_mut, dtype=np.uint8)
            expected = n * int(msg.point_step)
            if buf_all.size < expected:
                self.get_logger().warn("data size mismatch; skip frame")
                return
            rows = buf_all[:expected].reshape(n, int(msg.point_step))
            rows_kept = rows[keep].copy()
            n2 = int(rows_kept.shape[0])

            out = PointCloud2()
            out.header = msg.header
            out.height = 1
            out.width = n2
            out.fields = msg.fields
            out.is_bigendian = msg.is_bigendian
            out.point_step = msg.point_step
            out.row_step = int(n2 * msg.point_step)
            out.is_dense = msg.is_dense
            out.data = rows_kept.tobytes()
        else:
            out = PointCloud2()
            out.header = msg.header
            out.height = msg.height
            out.width = msg.width
            out.fields = msg.fields
            out.is_bigendian = msg.is_bigendian
            out.point_step = msg.point_step
            out.row_step = msg.row_step
            out.is_dense = msg.is_dense
            out.data = data_mut

        self.pub_pc.publish(out)

        # ---- publish GT + meta ----
        drop_ratio = float(np.mean(~keep)) if enable_drop else 0.0
        self._publish_gt_mask(msg.header, gt_mask)

        m1 = Float32()
        m1.data = float(drop_ratio)
        self.pub_drop.publish(m1)

        m2 = Float32()
        m2.data = float(bias_m if enable_spoof else 0.0)
        self.pub_bias.publish(m2)

        t1 = time.perf_counter_ns()
        if (t1 - t0) / 1e6 > 50.0:
            self.get_logger().warn(f"Perturber slow: {(t1 - t0)/1e6:.1f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPerturber()

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
