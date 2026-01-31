#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kitti_roi_eval/roi_frame_logger.py
#
# Lightweight frame logger for Chap7:
# - subscribe to frame_idx + ROI/GT masks + importance/reliability maps
# - save per-frame arrays/images for offline overlay rendering

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


def qos_sensor(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(max(1, depth)),
    )


def _try_save_image(path: str, arr_u8: np.ndarray) -> bool:
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr_u8)
        return True
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        if arr_u8.ndim == 2:
            plt.imsave(path, arr_u8, cmap="gray")
        else:
            plt.imsave(path, arr_u8)
        return True
    except Exception:
        return False


def _normalize_to_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)
    vmin = float(np.nanmin(x))
    vmax = float(np.nanmax(x))
    if vmax <= vmin + 1e-9:
        vmax = vmin + 1.0
    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


class RoiFrameLogger(Node):
    def __init__(self):
        super().__init__("roi_frame_logger")

        # ---- params ----
        self.declare_parameter("out_dir", "")
        self.declare_parameter("frame_idx_topic", "kitti_player/frame_idx")

        # save control (compatible with old yaml naming)
        self.declare_parameter("save_enable", True)
        self.declare_parameter("save_dir", "")
        self.declare_parameter("save_every_n", 1)
        self.declare_parameter("save_start_frame", 0)
        self.declare_parameter("save_stop_frame", 0)
        self.declare_parameter("save_format", "png")

        self.declare_parameter("save_importance", True)
        self.declare_parameter("save_reliability", False)
        self.declare_parameter("save_roi_rgb", True)
        self.declare_parameter("save_roi_masks", True)
        self.declare_parameter("save_omega", False)
        self.declare_parameter("save_rel_low", False)
        self.declare_parameter("save_gt_mask", False)
        self.declare_parameter("save_npy", True)
        self.declare_parameter("save_range", False)  # ignored (compat)
        self.declare_parameter("gt_npz_path", "")
        self.declare_parameter("split_masks_by_type", True)
        # visualization alignment
        self.declare_parameter("save_vis_aligned", True)
        self.declare_parameter("vis_dir", "vis")
        self.declare_parameter("vis_flip_ud", True)
        self.declare_parameter("vis_flip_lr", True)
        self.declare_parameter("vis_stretch", True)
        self.declare_parameter("vis_hfov_deg", 360.0)
        self.declare_parameter("vis_vfov_deg", 26.8)
        self.declare_parameter("vis_aspect_target", 3.312)  # KITTI image ~1242/375
        self.declare_parameter("vis_center_az", True)

        # topics
        self.declare_parameter("importance_topic", "roi_est/importance_map")
        self.declare_parameter("rel_topic", "roi_est/rel_map")
        self.declare_parameter("roi_imp_topic", "roi_est/roi_imp_mono8")
        self.declare_parameter("roi_use_topic", "roi_est/roi_use_mono8")
        self.declare_parameter("roi_alert_topic", "roi_est/roi_alert_mono8")
        self.declare_parameter("omega_topic", "roi_est/omega_mono8")
        self.declare_parameter("rel_low_topic", "roi_est/rel_low_mono8")
        self.declare_parameter("gt_mask_topic", "pc_perturber/gt_mask_mono8")

        # fallback shape (if MultiArrayLayout missing)
        self.declare_parameter("num_vertical_bins", 128)
        self.declare_parameter("num_horizontal_bins", 128)

        # ---- read params ----
        out_dir = str(self.get_parameter("out_dir").value).strip()
        save_dir = str(self.get_parameter("save_dir").value).strip()
        if out_dir == "" and save_dir != "":
            out_dir = save_dir
        if out_dir == "":
            out_dir = os.path.abspath("result_kitti_frames")
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.save_enable = bool(self.get_parameter("save_enable").value)
        self.save_every_n = int(self.get_parameter("save_every_n").value)
        self.save_start = int(self.get_parameter("save_start_frame").value)
        self.save_stop = int(self.get_parameter("save_stop_frame").value)
        self.save_format = str(self.get_parameter("save_format").value).lower()

        self.save_importance = bool(self.get_parameter("save_importance").value)
        self.save_reliability = bool(self.get_parameter("save_reliability").value)
        self.save_roi_rgb = bool(self.get_parameter("save_roi_rgb").value)
        self.save_roi_masks = bool(self.get_parameter("save_roi_masks").value)
        self.save_omega = bool(self.get_parameter("save_omega").value)
        self.save_rel_low = bool(self.get_parameter("save_rel_low").value)
        self.save_gt_mask = bool(self.get_parameter("save_gt_mask").value)
        self.save_npy = bool(self.get_parameter("save_npy").value)
        self.gt_npz_path = os.path.expanduser(str(self.get_parameter("gt_npz_path").value).strip())
        self.split_masks_by_type = bool(self.get_parameter("split_masks_by_type").value)
        self.save_vis_aligned = bool(self.get_parameter("save_vis_aligned").value)
        self.vis_dir = str(self.get_parameter("vis_dir").value).strip() or "vis"
        self.vis_flip_ud = bool(self.get_parameter("vis_flip_ud").value)
        self.vis_flip_lr = bool(self.get_parameter("vis_flip_lr").value)
        self.vis_stretch = bool(self.get_parameter("vis_stretch").value)
        self.vis_hfov_deg = float(self.get_parameter("vis_hfov_deg").value)
        self.vis_vfov_deg = float(self.get_parameter("vis_vfov_deg").value)
        self.vis_center_az = bool(self.get_parameter("vis_center_az").value)
        self.vis_aspect_target = float(self.get_parameter("vis_aspect_target").value)

        self.V = int(self.get_parameter("num_vertical_bins").value)
        self.H = int(self.get_parameter("num_horizontal_bins").value)

        # topic names
        self.frame_idx_topic = str(self.get_parameter("frame_idx_topic").value)
        self.importance_topic = str(self.get_parameter("importance_topic").value)
        self.rel_topic = str(self.get_parameter("rel_topic").value)
        self.roi_imp_topic = str(self.get_parameter("roi_imp_topic").value)
        self.roi_use_topic = str(self.get_parameter("roi_use_topic").value)
        self.roi_alert_topic = str(self.get_parameter("roi_alert_topic").value)
        self.omega_topic = str(self.get_parameter("omega_topic").value)
        self.rel_low_topic = str(self.get_parameter("rel_low_topic").value)
        self.gt_mask_topic = str(self.get_parameter("gt_mask_topic").value)

        # ---- directories ----
        self.dir_maps = os.path.join(self.out_dir, "maps")
        self.dir_masks_base = os.path.join(self.out_dir, "masks")
        self.dir_rgb = os.path.join(self.out_dir, "rgb")
        self.dir_vis = os.path.join(self.out_dir, self.vis_dir)
        self.dir_vis_maps = os.path.join(self.dir_vis, "maps")
        self.dir_vis_masks = os.path.join(self.dir_vis, "masks")
        self.dir_vis_rgb = os.path.join(self.dir_vis, "rgb")
        os.makedirs(self.dir_maps, exist_ok=True)
        os.makedirs(self.dir_masks_base, exist_ok=True)
        os.makedirs(self.dir_rgb, exist_ok=True)
        if self.save_vis_aligned:
            os.makedirs(self.dir_vis_maps, exist_ok=True)
            os.makedirs(self.dir_vis_masks, exist_ok=True)
            os.makedirs(self.dir_vis_rgb, exist_ok=True)

        # ---- state ----
        self.cur_frame: Optional[int] = None
        self._roi_use_cache: Optional[np.ndarray] = None
        self._roi_alert_cache: Optional[np.ndarray] = None
        self._roi_use_frame: Optional[int] = None
        self._roi_alert_frame: Optional[int] = None
        self._gt_map = {}
        self._gt_stack = None
        self._gt_saved = set()

        if self.save_gt_mask and self.gt_npz_path:
            try:
                z = np.load(self.gt_npz_path, allow_pickle=True)
                self._gt_stack = z["gt"].astype(np.uint8)
                frames = z.get("frame_indices", None)
                if frames is None:
                    start = int(z.get("start", 0))
                    frames = np.arange(start, start + self._gt_stack.shape[0], dtype=np.int32)
                frames = frames.astype(int)
                self._gt_map = {int(f): i for i, f in enumerate(frames)}
                self.get_logger().info(
                    f"  gt_npz_path={self.gt_npz_path} (frames={len(frames)})"
                )
            except Exception as e:
                self.get_logger().warn(f"gt_npz_path load failed: {self.gt_npz_path} ({e})")
                self._gt_map = {}
                self._gt_stack = None

        qos = qos_sensor(10)
        self.sub_idx = self.create_subscription(Int32, self.frame_idx_topic, self._cb_frame_idx, qos)

        self.sub_imp = self.create_subscription(Float32MultiArray, self.importance_topic, self._cb_importance, qos)
        self.sub_rel = self.create_subscription(Float32MultiArray, self.rel_topic, self._cb_reliability, qos)

        self.sub_imp_mask = self.create_subscription(Image, self.roi_imp_topic, self._cb_roi_imp, qos)
        self.sub_use_mask = self.create_subscription(Image, self.roi_use_topic, self._cb_roi_use, qos)
        self.sub_alert_mask = self.create_subscription(Image, self.roi_alert_topic, self._cb_roi_alert, qos)
        self.sub_omega_mask = self.create_subscription(Image, self.omega_topic, self._cb_omega, qos)
        self.sub_rel_low_mask = self.create_subscription(Image, self.rel_low_topic, self._cb_rel_low, qos)
        self.sub_gt_mask = self.create_subscription(Image, self.gt_mask_topic, self._cb_gt_mask, qos)

        self.get_logger().info(
            "RoiFrameLogger started.\n"
            f"  out_dir={self.out_dir}\n"
            f"  save_enable={self.save_enable} every_n={self.save_every_n} range={self.save_start}..{self.save_stop}\n"
            f"  topics: frame_idx={self.frame_idx_topic}, imp={self.importance_topic}, rel={self.rel_topic}\n"
        )

    def _cb_frame_idx(self, msg: Int32):
        self.cur_frame = int(msg.data)
        if self.save_gt_mask and self._gt_stack is not None:
            fi = self.cur_frame
            if self._should_save(fi) and fi not in self._gt_saved:
                gi = self._gt_map.get(fi, None)
                if gi is not None:
                    gt = self._gt_stack[int(gi)]
                    self._save_mask("gt_mask", (gt > 0).astype(np.uint8) * 255, fi)
                    self._gt_saved.add(fi)

    def _should_save(self, frame_idx: Optional[int]) -> bool:
        if not self.save_enable:
            return False
        if frame_idx is None:
            return False
        if frame_idx < self.save_start:
            return False
        if self.save_stop > 0 and frame_idx > self.save_stop:
            return False
        if self.save_every_n > 1 and (frame_idx % self.save_every_n) != 0:
            return False
        return True

    def _ma_to_2d(self, msg: Float32MultiArray) -> Optional[np.ndarray]:
        try:
            if msg.layout.dim and len(msg.layout.dim) >= 2:
                v = int(msg.layout.dim[0].size)
                h = int(msg.layout.dim[1].size)
            else:
                v = int(self.V)
                h = int(self.H)
            arr = np.array(msg.data, dtype=np.float32)
            if arr.size != v * h:
                return None
            return arr.reshape(v, h)
        except Exception:
            return None

    def _resize_nn(self, arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        if out_h <= 0 or out_w <= 0:
            return arr
        in_h, in_w = arr.shape[:2]
        if in_h == out_h and in_w == out_w:
            return arr
        y_idx = (np.linspace(0, in_h - 1, out_h)).round().astype(np.int32)
        x_idx = (np.linspace(0, in_w - 1, out_w)).round().astype(np.int32)
        if arr.ndim == 2:
            return arr[y_idx[:, None], x_idx[None, :]]
        return arr[y_idx[:, None], x_idx[None, :], :]

    def _align_vis(self, arr: np.ndarray) -> np.ndarray:
        out = arr
        if self.vis_flip_ud:
            out = out[::-1, ...]
        if self.vis_flip_lr:
            out = out[:, ::-1, ...]
        if self.vis_center_az:
            # roll horizontally by half width to center az=0
            w = out.shape[1]
            shift = int(w // 2)
            out = np.roll(out, shift=shift, axis=1)
        if self.vis_stretch:
            h, w = out.shape[:2]
            aspect_des = float(self.vis_aspect_target)
            aspect_cur = float(w) / max(1e-6, float(h))
            scale_w = aspect_des / max(1e-6, aspect_cur)
            target_w = max(1, int(round(w * scale_w)))
            out = self._resize_nn(out, h, target_w)
        return out

    def _save_map(self, name: str, arr: np.ndarray, frame_idx: int):
        if self.save_npy:
            npy_path = os.path.join(self.dir_maps, f"{name}_{frame_idx:06d}.npy")
            np.save(npy_path, arr.astype(np.float32))

        if self.save_format in ("png", "pgm"):
            img = _normalize_to_u8(arr)
            ext = "png" if self.save_format == "png" else "pgm"
            img_path = os.path.join(self.dir_maps, f"{name}_{frame_idx:06d}.{ext}")
            if not _try_save_image(img_path, img):
                np.save(img_path + ".npy", img)
            if self.save_vis_aligned:
                vis = self._align_vis(img)
                vis_path = os.path.join(self.dir_vis_maps, f"{name}_{frame_idx:06d}.{ext}")
                if not _try_save_image(vis_path, vis):
                    np.save(vis_path + ".npy", vis)

    def _save_mask(self, name: str, mask_u8: np.ndarray, frame_idx: int):
        if self.split_masks_by_type:
            base_dir = os.path.join(self.dir_masks_base, name)
        else:
            base_dir = self.dir_masks_base
        os.makedirs(base_dir, exist_ok=True)
        ext = "png" if self.save_format == "png" else "pgm"
        path = os.path.join(base_dir, f"{name}_{frame_idx:06d}.{ext}")
        if not _try_save_image(path, mask_u8):
            np.save(path + ".npy", mask_u8)
        if self.save_vis_aligned:
            vis = self._align_vis(mask_u8)
            vis_dir = os.path.join(self.dir_vis_masks, name) if self.split_masks_by_type else self.dir_vis_masks
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{name}_{frame_idx:06d}.{ext}")
            if not _try_save_image(vis_path, vis):
                np.save(vis_path + ".npy", vis)

    def _save_roi_rgb(self, frame_idx: int, roi_use: np.ndarray, roi_alert: np.ndarray):
        rgb = np.zeros((roi_use.shape[0], roi_use.shape[1], 3), dtype=np.uint8)
        rgb[roi_use > 0, 1] = 255
        rgb[roi_alert > 0, 0] = 255
        path = os.path.join(self.dir_rgb, f"roi_rgb_{frame_idx:06d}.png")
        if not _try_save_image(path, rgb):
            np.save(path + ".npy", rgb)
        if self.save_vis_aligned:
            vis = self._align_vis(rgb)
            vis_path = os.path.join(self.dir_vis_rgb, f"roi_rgb_{frame_idx:06d}.png")
            if not _try_save_image(vis_path, vis):
                np.save(vis_path + ".npy", vis)

    def _cb_importance(self, msg: Float32MultiArray):
        if not self.save_importance:
            return
        frame_idx = self.cur_frame
        if not self._should_save(frame_idx):
            return
        arr = self._ma_to_2d(msg)
        if arr is None:
            return
        self._save_map("importance", arr, frame_idx)

    def _cb_reliability(self, msg: Float32MultiArray):
        if not self.save_reliability:
            return
        frame_idx = self.cur_frame
        if not self._should_save(frame_idx):
            return
        arr = self._ma_to_2d(msg)
        if arr is None:
            return
        self._save_map("reliability", arr, frame_idx)

    def _cb_roi_imp(self, msg: Image):
        if not self.save_roi_masks:
            return
        frame_idx = self.cur_frame
        if not self._should_save(frame_idx):
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
        self._save_mask("roi_imp", a, frame_idx)

    def _cb_roi_use(self, msg: Image):
        frame_idx = self.cur_frame
        if frame_idx is None:
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))

        if self.save_roi_masks and self._should_save(frame_idx):
            self._save_mask("roi_use", a, frame_idx)

        if self.save_roi_rgb:
            self._roi_use_cache = a
            self._roi_use_frame = frame_idx
            if self._roi_alert_cache is not None and self._roi_alert_frame == frame_idx:
                if self._should_save(frame_idx):
                    self._save_roi_rgb(frame_idx, self._roi_use_cache, self._roi_alert_cache)

    def _cb_roi_alert(self, msg: Image):
        frame_idx = self.cur_frame
        if frame_idx is None:
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))

        if self.save_roi_masks and self._should_save(frame_idx):
            self._save_mask("roi_alert", a, frame_idx)

        if self.save_roi_rgb:
            self._roi_alert_cache = a
            self._roi_alert_frame = frame_idx
            if self._roi_use_cache is not None and self._roi_use_frame == frame_idx:
                if self._should_save(frame_idx):
                    self._save_roi_rgb(frame_idx, self._roi_use_cache, self._roi_alert_cache)

    def _cb_omega(self, msg: Image):
        if not self.save_omega:
            return
        frame_idx = self.cur_frame
        if not self._should_save(frame_idx):
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
        self._save_mask("omega", a, frame_idx)

    def _cb_rel_low(self, msg: Image):
        if not self.save_rel_low:
            return
        frame_idx = self.cur_frame
        if not self._should_save(frame_idx):
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
        self._save_mask("rel_low", a, frame_idx)

    def _cb_gt_mask(self, msg: Image):
        if not self.save_gt_mask:
            return
        frame_idx = self.cur_frame
        if not self._should_save(frame_idx):
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.width))
        self._save_mask("gt_mask", a, frame_idx)


def main(args=None):
    rclpy.init(args=args)
    node = RoiFrameLogger()
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
