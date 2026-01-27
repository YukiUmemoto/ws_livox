#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Header, Int32
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


def make_qos(depth: int = 50, reliable: bool = True) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


class KittiPlayerFinishFirst(Node):
    """
    完走優先（finish-first）
    - frame_idx(Int32) を publish（これを唯一の同期キーにする）
    - 点群(PointCloud2) と points_count(Int32) を publish
    - logger から ack_frame_idx(Int32) が返るまで同一フレームを再送
    """

    def __init__(self):
        super().__init__("kitti_player_finish_first")

        # ---- parameters (launch 互換) ----
        self.declare_parameter("drive_dir", "")
        self.declare_parameter("kitti_drive_dir", "")  # alias

        self.declare_parameter("points_topic", "/livox/lidar_perturbed")
        self.declare_parameter("frame_idx_topic", "kitti_player/frame_idx")
        self.declare_parameter("points_count_topic", "kitti_player/points_count")

        self.declare_parameter("start_idx", 0)
        self.declare_parameter("end_idx", -1)
        self.declare_parameter("stride", 1)

        self.declare_parameter("finish_first", True)
        self.declare_parameter("ack_topic", "kitti_player/ack_frame_idx")
        self.declare_parameter("ack_timeout_sec", 120.0)
        self.declare_parameter("republish_interval_sec", 1.0)

        # QoS（重要：点群は RELIABLE 推奨。subscriber(best_effort)とも互換が取りやすい）
        self.declare_parameter("qos_depth", 50)
        self.declare_parameter("qos_points_reliable", True)
        self.declare_parameter("qos_ack_reliable", True)

        self.declare_parameter("frame_id", "velo")

        # ---- read ----
        drive_dir = str(self.get_parameter("drive_dir").value).strip()
        if drive_dir == "":
            drive_dir = str(self.get_parameter("kitti_drive_dir").value).strip()
        if drive_dir == "":
            raise RuntimeError("drive_dir is empty.")
        self.drive_dir = os.path.expanduser(drive_dir)

        self.points_topic = str(self.get_parameter("points_topic").value)
        self.frame_idx_topic = str(self.get_parameter("frame_idx_topic").value)
        self.points_count_topic = str(self.get_parameter("points_count_topic").value)

        self.start_idx = int(self.get_parameter("start_idx").value)
        self.end_idx = int(self.get_parameter("end_idx").value)
        self.stride = int(self.get_parameter("stride").value)

        self.finish_first = bool(self.get_parameter("finish_first").value)
        self.ack_topic = str(self.get_parameter("ack_topic").value)
        self.ack_timeout_sec = float(self.get_parameter("ack_timeout_sec").value)
        self.republish_interval_sec = float(self.get_parameter("republish_interval_sec").value)

        qos_depth = int(self.get_parameter("qos_depth").value)
        qos_points_reliable = bool(self.get_parameter("qos_points_reliable").value)
        qos_ack_reliable = bool(self.get_parameter("qos_ack_reliable").value)

        self.qos_points = make_qos(depth=qos_depth, reliable=qos_points_reliable)
        self.qos_ack = make_qos(depth=max(10, qos_depth), reliable=qos_ack_reliable)

        self.frame_id = str(self.get_parameter("frame_id").value)

        # ---- locate data ----
        velo_dir = os.path.join(self.drive_dir, "velodyne_points", "data")
        self.point_files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
        if len(self.point_files) == 0:
            raise FileNotFoundError(f"no velodyne .bin found: {velo_dir}")

        last_idx = len(self.point_files) - 1
        if self.end_idx < 0:
            self.end_idx = last_idx
        self.end_idx = min(self.end_idx, last_idx)
        self.start_idx = max(0, min(self.start_idx, self.end_idx))

        # ---- pubs/subs ----
        self.pub_pc = self.create_publisher(PointCloud2, self.points_topic, self.qos_points)
        self.pub_frame_idx = self.create_publisher(Int32, self.frame_idx_topic, self.qos_points)
        self.pub_points_count = self.create_publisher(Int32, self.points_count_topic, self.qos_points)

        self._ack_lock = threading.Condition()
        self._last_ack = -1
        self.sub_ack = self.create_subscription(Int32, self.ack_topic, self._cb_ack, self.qos_ack)

        self._stop = False
        self.cur = int(self.start_idx)

        self.get_logger().info(
            "KittiPlayerFinishFirst started.\n"
            f"  drive_dir={self.drive_dir}\n"
            f"  frames(total)={len(self.point_files)} start={self.start_idx} end={self.end_idx} stride={self.stride}\n"
            f"  points_topic={self.points_topic}\n"
            f"  frame_idx_topic={self.frame_idx_topic}\n"
            f"  points_count_topic={self.points_count_topic}\n"
            f"  finish_first={self.finish_first} ack_topic={self.ack_topic} ack_timeout_sec={self.ack_timeout_sec}\n"
            f"  republish_interval_sec={self.republish_interval_sec}\n"
            f"  qos_points: reliable={qos_points_reliable} depth={qos_depth}\n"
            f"  qos_ack:    reliable={qos_ack_reliable} depth={max(10, qos_depth)}\n"
        )

        self.worker = threading.Thread(target=self._run_finish_first, daemon=True)
        self.worker.start()

    def _cb_ack(self, msg: Int32):
        with self._ack_lock:
            self._last_ack = max(self._last_ack, int(msg.data))
            self._ack_lock.notify_all()

    def _make_header(self) -> Header:
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.frame_id
        return h

    def _make_pc2(self, header: Header, xyz: np.ndarray) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        return point_cloud2.create_cloud(header, fields, xyz.astype(np.float32))

    def _publish_one(self, frame_idx: int):
        path = self.point_files[frame_idx]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]
        npts = int(xyz.shape[0])

        # frame_idx を「先に」投げる
        m_idx = Int32()
        m_idx.data = int(frame_idx)
        self.pub_frame_idx.publish(m_idx)

        m_cnt = Int32()
        m_cnt.data = int(npts)
        self.pub_points_count.publish(m_cnt)

        header = self._make_header()
        pc2 = self._make_pc2(header, xyz)
        self.pub_pc.publish(pc2)

    def _wait_ack_with_republish(self, frame_idx: int) -> bool:
        t0 = time.monotonic()
        next_pub = t0  # すぐ1回 publish したい
        while rclpy.ok() and (time.monotonic() - t0) < self.ack_timeout_sec and not self._stop:
            # 既に ack 済みなら終了
            with self._ack_lock:
                if self._last_ack >= frame_idx:
                    return True

            now = time.monotonic()
            if now >= next_pub:
                self._publish_one(frame_idx)
                next_pub = now + max(0.05, self.republish_interval_sec)

            # 条件変数で少し待つ（busy loop 回避）
            with self._ack_lock:
                self._ack_lock.wait(timeout=0.05)

        with self._ack_lock:
            return bool(self._last_ack >= frame_idx)

    def _run_finish_first(self):
        self.get_logger().info("finish-first worker started.")

        # subscriber が付くまで少し待つ（起動直後ドロップ対策）
        # ここは必須ではないが、最初の1枚落ちを減らす。
        for _ in range(50):
            if self.pub_pc.get_subscription_count() > 0 and self.pub_frame_idx.get_subscription_count() > 0:
                break
            time.sleep(0.1)

        while rclpy.ok() and not self._stop and self.cur <= self.end_idx:
            frame_idx = int(self.cur)

            ok = self._wait_ack_with_republish(frame_idx)
            if not ok:
                self.get_logger().error(f"ACK timeout at frame_idx={frame_idx}. stop.")
                break

            if (frame_idx % 10) == 0:
                self.get_logger().info(f"acked frame_idx={frame_idx}")

            self.cur += max(1, int(self.stride))

        self.get_logger().info("finish-first done.")
        self._stop = True


def main():
    rclpy.init()
    node = KittiPlayerFinishFirst()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
