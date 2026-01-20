# kitti_roi_eval/kitti_calib.py
from __future__ import annotations
import numpy as np

def load_calib_velo_to_cam(calib_velo_to_cam_txt: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (R:3x3, T:3,)
    calib_velo_to_cam.txt の典型形式:
      R: r11 r12 ... r33
      T: t1 t2 t3
    """
    R = None
    T = None
    with open(calib_velo_to_cam_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("R:"):
                vals = [float(x) for x in line.split()[1:]]
                R = np.array(vals, dtype=np.float32).reshape(3, 3)
            elif line.startswith("T:"):
                vals = [float(x) for x in line.split()[1:]]
                T = np.array(vals, dtype=np.float32).reshape(3,)
    if R is None or T is None:
        raise RuntimeError(f"Failed to parse R/T from {calib_velo_to_cam_txt}")
    return R, T

def transform_velo_to_cam(points_velo: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    # points: Nx3
    return (points_velo @ R.T) + T.reshape(1, 3)
