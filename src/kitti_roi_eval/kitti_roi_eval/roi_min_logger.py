#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/roi_min_logger.py
#
# Finish-first logger:
# - write one CSV row per dataset frame (guaranteed on done)
# - associate messages by simulated stamp -> frame_idx
# - no GT, no visualization (offline later)

from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Image, PointCloud2


def qos_sensor(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def qos_reliable(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def img_to_bool(msg: Image) -> np.ndarray:
    if msg.encoding != "mono8":
        raise RuntimeError(f"Expected mono8, got {msg.encoding}")
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
    return (a > 0)


def stamp_to_float_s(sec: int, nsec: int) -> float:
    return float(sec) + float(nsec) * 1e-9


@dataclass
class Entry:
    frame_idx: int
    stamp_sec: int = -1
    stamp_nsec: int = -1
    points_count: int = -1
    pred_bins: int = -1
    omega_bins: int = -1
    proc_time_ms: float = -1.0
    total_bins: int = -1

    pc_missing: int = 1
    pred_missing: int = 1
    omega_missing: int = 1
    proc_missing: int = 1

    written: bool = False


class RoiMinLogger(Node):
    """
    Realtime minimal logger:
      - subscribes to pc/pred/omega/proc_time and done
      - maps each msg to frame_idx using stamp (assumes sim stamp is used)
      - guarantees one line per frame_idx in [start_idx, end_idx] when done is received
    """

    def __init__(self):
        super().__init__("roi_min_logger")

        # ---- params ----
        self.declare_parameter("out_dir", "")
        self.declare_parameter("csv_name", "min_log.csv")

        self.declare_parameter("done_topic", "kitti_player/done")

        self.declare_parameter("points_topic", "/livox/lidar_perturbed")
        self.declare_parameter("pred_topic", "roi_est/roi_imp_mono8")
        self.declare_parameter("omega_topic", "roi_est/omega_mono8")
        self.declare_parameter("proc_time_topic", "roi_est/proc_time_ms")  # optional

        self.declare_parameter("rate_hz", 10.0)  # must match player rate_hz for sim stamp
        self.declare_parameter("start_idx", 0)
        self.declare_parameter("end_idx", 232)   # inclusive (233 files -> 0..232)

        self.declare_parameter("num_vertical_bins", 128)
        self.declare_parameter("num_horizontal_bins", 128)

        self.declare_parameter("qos_depth", 10)

        # ---- read params ----
        out_dir = str(self.get_parameter("out_dir").value).strip()
        if out_dir == "":
            out_dir = os.path.abspath("result_kitti_minlog")
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, str(self.get_parameter("csv_name").value))
        self.done_topic = str(self.get_parameter("done_topic").value)

        self.points_topic = str(self.get_parameter("points_topic").value)
        self.pred_topic = str(self.get_parameter("pred_topic").value)
        self.omega_topic = str(self.get_parameter("omega_topic").value)
        self.proc_time_topic = str(self.get_parameter("proc_time_topic").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.start_idx = int(self.get_parameter("start_idx").value)
        self.end_idx = int(self.get_parameter("end_idx").value)

        self.V = int(self.get_parameter("num_vertical_bins").value)
        self.H = int(self.get_parameter("num_horizontal_bins").value)
        self.total_bins = int(self.V) * int(self.H)

        qos_depth = int(self.get_parameter("qos_depth").value)

        # ---- state ----
        self.entries: Dict[int, Entry] = {}
        self.done_received = False

        # pre-create entries to guarantee full coverage on done
        for i in range(self.start_idx, self.end_idx + 1):
            self.entries[i] = Entry(frame_idx=i, total_bins=self.total_bins)

        # ---- CSV open ----
        self._fp = open(self.csv_path, "w", newline="")
        self._wr = csv.writer(self._fp)
        self._wr.writerow([
            "frame_idx",
            "stamp_sec",
            "stamp_nanosec",
            "points_count",
            "total_bins",
            "pred_bins",
            "omega_bins",
            "proc_time_ms",
            "pc_missing",
            "pred_missing",
            "omega_missing",
            "proc_missing",
        ])
        self._fp.flush()

        # ---- subs ----
        self.sub_done = self.create_subscription(Bool, self.done_topic, self._cb_done, qos_reliable(qos_depth))

        # heavy topics: best_effort
        self.sub_pc = self.create_subscription(PointCloud2, self.points_topic, self._cb_pc, qos_sensor(qos_depth))
        self.sub_pred = self.create_subscription(Image, self.pred_topic, self._cb_pred, qos_sensor(qos_depth))
        self.sub_omega = self.create_subscription(Image, self.omega_topic, self._cb_omega, qos_sensor(qos_depth))

        # optional
        self.sub_pt = self.create_subscription(Float32, self.proc_time_topic, self._cb_proc_time, qos_sensor(qos_depth))

        self.get_logger().info(
            "RoiMinLogger started.\n"
            f"  out_dir={self.out_dir}\n"
            f"  csv={self.csv_path}\n"
            f"  range={self.start_idx}..{self.end_idx}\n"
            f"  rate_hz={self.rate_hz} (must match sim stamp)\n"
            f"  topics: pc={self.points_topic}, pred={self.pred_topic}, omega={self.omega_topic}, proc={self.proc_time_topic}, done={self.done_topic}\n"
            f"  total_bins={self.total_bins} (VxH={self.V}x{self.H})\n"
        )

    def _frame_idx_from_stamp(self, sec: int, nsec: int) -> int:
        # frame_idx = round(t * rate_hz)
        t = stamp_to_float_s(sec, nsec)
        return int(round(t * self.rate_hz))

    def _get_entry(self, idx: int) -> Optional[Entry]:
        return self.entries.get(idx)

    def _maybe_write(self, e: Entry):
        # ここでは即時書き込みはしない（重複/順序を避けるため）
        # done受信時にまとめて確実に書く方針
        pass

    def _cb_done(self, msg: Bool):
        if not bool(msg.data):
            return
        if self.done_received:
            return
        self.done_received = True
        self.get_logger().info("done received -> flush all frames to CSV.")
        self._flush_all_and_close()

    def _cb_pc(self, msg: PointCloud2):
        sec = int(msg.header.stamp.sec)
        nsec = int(msg.header.stamp.nanosec)
        idx = self._frame_idx_from_stamp(sec, nsec)
        e = self._get_entry(idx)
        if e is None:
            return
        e.stamp_sec = sec
        e.stamp_nsec = nsec
        e.points_count = int(msg.width) * int(msg.height)
        e.pc_missing = 0

    def _cb_pred(self, msg: Image):
        sec = int(msg.header.stamp.sec)
        nsec = int(msg.header.stamp.nanosec)
        idx = self._frame_idx_from_stamp(sec, nsec)
        e = self._get_entry(idx)
        if e is None:
            return
        try:
            m = img_to_bool(msg)
        except Exception:
            return
        e.stamp_sec = sec
        e.stamp_nsec = nsec
        e.pred_bins = int(np.sum(m))
        e.pred_missing = 0

    def _cb_omega(self, msg: Image):
        sec = int(msg.header.stamp.sec)
        nsec = int(msg.header.stamp.nanosec)
        idx = self._frame_idx_from_stamp(sec, nsec)
        e = self._get_entry(idx)
        if e is None:
            return
        try:
            m = img_to_bool(msg)
        except Exception:
            return
        e.stamp_sec = sec
        e.stamp_nsec = nsec
        e.omega_bins = int(np.sum(m))
        e.omega_missing = 0

    def _cb_proc_time(self, msg: Float32):
        # proc_time は stamp が無いので「最後に更新されたフレームに紐づく」方式だと危険
        # → 可能なら推定器側で Header付き msg にするのが理想
        #
        # ただし「完走優先」で最低限埋める用途として、ここでは最新に近いフレームへ近似で入れます。
        # オフライン評価で正確に詰めるなら、推定器側に stamp付きproc_timeを出させてください。
        v = float(msg.data)

        # 近似：未埋めの中で最小idxのproc_missingを埋める（崩れにくい）
        for i in range(self.start_idx, self.end_idx + 1):
            e = self.entries[i]
            if e.proc_missing == 1:
                e.proc_time_ms = v
                e.proc_missing = 0
                break

    def _flush_all_and_close(self):
        # 必ず start..end の全行を出す
        for i in range(self.start_idx, self.end_idx + 1):
            e = self.entries[i]
            if e.written:
                continue
            row = [
                int(e.frame_idx),
                int(e.stamp_sec),
                int(e.stamp_nsec),
                int(e.points_count),
                int(e.total_bins),
                int(e.pred_bins),
                int(e.omega_bins),
                float(e.proc_time_ms),
                int(e.pc_missing),
                int(e.pred_missing),
                int(e.omega_missing),
                int(e.proc_missing),
            ]
            self._wr.writerow(row)
            e.written = True

        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass

        self.get_logger().info(f"[minlog] wrote: {self.csv_path}")

        # shutdown this node
        try:
            self.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        try:
            if hasattr(self, "_fp") and self._fp is not None:
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
        return super().destroy_node()


def main():
    rclpy.init()
    node = RoiMinLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
