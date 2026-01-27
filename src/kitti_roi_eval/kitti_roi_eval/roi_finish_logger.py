#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import csv
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32
from sensor_msgs.msg import Image, PointCloud2


def make_qos(depth: int = 50, reliable: bool = False) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def img_to_bool(msg: Image) -> np.ndarray:
    if msg.encoding != "mono8":
        raise RuntimeError(f"Expected mono8, got {msg.encoding}")
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
    return (a > 0)


class RoiFinishLogger(Node):
    """
    完走優先ロガー
    - frame_idx_topic(Int32) を唯一のキーとし、header.stamp には依存しない
    - active_frame が 1 つだけ進む（finish-first 前提）
    - pred + omega + points_count(or pc) が揃ったら 1 行書いて ack を返す
    """

    def __init__(self):
        super().__init__("roi_finish_logger")

        self.declare_parameter("pred_topic", "roi_est/roi_imp_mono8")
        self.declare_parameter("omega_topic", "roi_est/omega_mono8")
        self.declare_parameter("pc_topic", "/livox/lidar_perturbed")

        self.declare_parameter("frame_idx_topic", "kitti_player/frame_idx")
        self.declare_parameter("points_count_topic", "kitti_player/points_count")  # あるなら使う

        self.declare_parameter("ack_topic", "kitti_player/ack_frame_idx")

        self.declare_parameter("out_dir", "")
        self.declare_parameter("csv_name", "min_log.csv")

        self.declare_parameter("qos_depth", 50)
        self.declare_parameter("qos_sub_reliable", False)  # estimator が BEST_EFFORT の想定
        self.declare_parameter("qos_ack_reliable", True)

        self.declare_parameter("debug_first_n", 3)

        # ---- read ----
        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.omega_topic = str(self.get_parameter("omega_topic").value)
        self.pc_topic = str(self.get_parameter("pc_topic").value)

        self.frame_idx_topic = str(self.get_parameter("frame_idx_topic").value)
        self.points_count_topic = str(self.get_parameter("points_count_topic").value)

        self.ack_topic = str(self.get_parameter("ack_topic").value)

        out_dir = str(self.get_parameter("out_dir").value).strip()
        if out_dir == "":
            out_dir = os.path.abspath("result_kitti_finish_first")
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, str(self.get_parameter("csv_name").value))

        qos_depth = int(self.get_parameter("qos_depth").value)
        qos_sub_reliable = bool(self.get_parameter("qos_sub_reliable").value)
        qos_ack_reliable = bool(self.get_parameter("qos_ack_reliable").value)

        self.qos_sub = make_qos(depth=qos_depth, reliable=qos_sub_reliable)
        self.qos_ack = make_qos(depth=max(10, qos_depth), reliable=qos_ack_reliable)

        self.debug_first_n = int(self.get_parameter("debug_first_n").value)

        # ---- state ----
        self.active_frame: Optional[int] = None
        self.done_max = -1

        self._pred: Optional[np.ndarray] = None
        self._omega: Optional[np.ndarray] = None
        self._points_count: Optional[int] = None

        self._t_frame_ns: Optional[int] = None
        self._t_pred_ns: Optional[int] = None
        self._t_omega_ns: Optional[int] = None
        self._t_pc_ns: Optional[int] = None
        self._t_cnt_ns: Optional[int] = None

        # ---- CSV ----
        self._fp = open(self.csv_path, "w", newline="")
        self._wr = csv.writer(self._fp)
        self._wr.writerow([
            "frame_idx",
            "pred_bins_on_omega",
            "omega_bins",
            "pred_ratio_on_omega",
            "points_in_frame",
            "total_bins",
            "lat_pred_ms_from_frame",
            "lat_omega_ms_from_frame",
        ])
        self._fp.flush()

        # ---- pubs/subs ----
        self.pub_ack = self.create_publisher(Int32, self.ack_topic, self.qos_ack)

        self.sub_frame = self.create_subscription(Int32, self.frame_idx_topic, self._cb_frame_idx, self.qos_sub)
        self.sub_pred = self.create_subscription(Image, self.pred_topic, self._cb_pred, self.qos_sub)
        self.sub_omega = self.create_subscription(Image, self.omega_topic, self._cb_omega, self.qos_sub)

        # points_count が来なければ pc から数える（どちらでも良い）
        self.sub_cnt = self.create_subscription(Int32, self.points_count_topic, self._cb_points_count, self.qos_sub)
        self.sub_pc = self.create_subscription(PointCloud2, self.pc_topic, self._cb_pc, self.qos_sub)

        self.get_logger().info(
            "RoiFinishLogger started.\n"
            f"  pred={self.pred_topic}\n"
            f"  omega={self.omega_topic}\n"
            f"  pc={self.pc_topic}\n"
            f"  frame_idx_topic={self.frame_idx_topic}\n"
            f"  points_count_topic={self.points_count_topic}\n"
            f"  ack_topic={self.ack_topic}\n"
            f"  out_dir={self.out_dir}\n"
            f"  csv={self.csv_path}\n"
            f"  qos_sub reliable={qos_sub_reliable} depth={qos_depth}\n"
            f"  qos_ack reliable={qos_ack_reliable} depth={max(10, qos_depth)}\n"
        )

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _reset_buffers_for_new_frame(self, frame_idx: int):
        self.active_frame = int(frame_idx)
        self._pred = None
        self._omega = None
        self._points_count = None
        self._t_frame_ns = self._now_ns()
        self._t_pred_ns = None
        self._t_omega_ns = None
        self._t_pc_ns = None
        self._t_cnt_ns = None

    def _cb_frame_idx(self, msg: Int32):
        fi = int(msg.data)

        # 既に完了したフレームなら即 ACK（再送対策）
        if fi <= self.done_max:
            ack = Int32()
            ack.data = fi
            self.pub_ack.publish(ack)
            return

        # 新しいフレームとしてセット
        if self.active_frame is None or fi != self.active_frame:
            self._reset_buffers_for_new_frame(fi)
            if fi < self.debug_first_n:
                self.get_logger().info(f"[dbg] got frame_idx={fi}")
        self._try_finalize()

    def _cb_points_count(self, msg: Int32):
        if self.active_frame is None:
            return
        self._points_count = int(msg.data)
        self._t_cnt_ns = self._now_ns()
        if self.active_frame < self.debug_first_n:
            self.get_logger().info(f"[dbg] got points_count={self._points_count} for frame={self.active_frame}")
        self._try_finalize()

    def _cb_pc(self, msg: PointCloud2):
        if self.active_frame is None:
            return
        # points_count が未到着なら pc から計算して埋める
        if self._points_count is None:
            self._points_count = int(msg.width) * int(msg.height)
        self._t_pc_ns = self._now_ns()
        if self.active_frame < self.debug_first_n:
            self.get_logger().info(f"[dbg] got pc for frame={self.active_frame} (npts={self._points_count})")
        self._try_finalize()

    def _cb_pred(self, msg: Image):
        if self.active_frame is None:
            return
        try:
            self._pred = img_to_bool(msg)
        except Exception as e:
            self.get_logger().error(f"pred parse failed: {e}")
            return
        self._t_pred_ns = self._now_ns()
        if self.active_frame < self.debug_first_n:
            self.get_logger().info(f"[dbg] got pred for frame={self.active_frame}")
        self._try_finalize()

    def _cb_omega(self, msg: Image):
        if self.active_frame is None:
            return
        try:
            self._omega = img_to_bool(msg)
        except Exception as e:
            self.get_logger().error(f"omega parse failed: {e}")
            return
        self._t_omega_ns = self._now_ns()
        if self.active_frame < self.debug_first_n:
            self.get_logger().info(f"[dbg] got omega for frame={self.active_frame}")
        self._try_finalize()

    def _try_finalize(self):
        if self.active_frame is None:
            return
        fi = int(self.active_frame)

        if fi <= self.done_max:
            return
        if self._pred is None or self._omega is None or self._points_count is None:
            return

        pred = self._pred
        omega = self._omega

        if pred.shape != omega.shape:
            self.get_logger().error(f"shape mismatch frame={fi}: pred={pred.shape}, omega={omega.shape}")
            return

        valid = omega.astype(bool)
        p_on = pred.astype(bool) & valid

        omega_bins = int(np.sum(valid))
        pred_bins_on_omega = int(np.sum(p_on))
        total_bins = int(pred.size)
        pred_ratio = float(pred_bins_on_omega / max(1, omega_bins))

        t0 = self._t_frame_ns if self._t_frame_ns is not None else self._now_ns()
        lat_pred_ms = float((self._t_pred_ns - t0) * 1e-6) if self._t_pred_ns is not None else -1.0
        lat_omega_ms = float((self._t_omega_ns - t0) * 1e-6) if self._t_omega_ns is not None else -1.0

        self._wr.writerow([
            fi,
            pred_bins_on_omega,
            omega_bins,
            pred_ratio,
            int(self._points_count),
            total_bins,
            lat_pred_ms,
            lat_omega_ms,
        ])
        self._fp.flush()

        self.done_max = max(self.done_max, fi)

        ack = Int32()
        ack.data = fi
        self.pub_ack.publish(ack)

        # 次フレームを待つ
        self.active_frame = None

        if (fi % 10) == 0:
            self.get_logger().info(
                f"[done] frame={fi} pred_on_omega={pred_bins_on_omega} omega={omega_bins} "
                f"ratio={pred_ratio:.3f} npts={int(self._points_count)}"
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
    node = RoiFinishLogger()
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
