# kitti_roi_eval/kitti_tracklet.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

@dataclass
class BoxPose:
    frame: int
    tx: float
    ty: float
    tz: float
    rz: float  # yaw候補（使う軸は上位で選択）
    rx: float = 0.0
    ry: float = 0.0

@dataclass
class Tracklet:
    obj_type: str
    h: float
    w: float
    l: float
    first_frame: int
    poses: List[BoxPose]

def _get_text(node: ET.Element, tag: str, default: str = "") -> str:
    x = node.find(tag)
    return x.text.strip() if (x is not None and x.text is not None) else default

def _get_float(node: ET.Element, tag: str, default: float = 0.0) -> float:
    s = _get_text(node, tag, "")
    try:
        return float(s)
    except Exception:
        return default

def _get_int(node: ET.Element, tag: str, default: int = 0) -> int:
    s = _get_text(node, tag, "")
    try:
        return int(float(s))
    except Exception:
        return default

def parse_tracklet_labels(xml_path: str) -> List[Tracklet]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # KITTI raw tracklets は root が tracklets の場合が多い
    tracklets_node = root if root.tag == "tracklets" else root.find("tracklets")
    if tracklets_node is None:
        raise RuntimeError(f"tracklets node not found in: {xml_path}")

    out: List[Tracklet] = []
    for t in tracklets_node.findall("item"):
        obj_type = _get_text(t, "objectType", "")
        h = _get_float(t, "h", 0.0)
        w = _get_float(t, "w", 0.0)
        l = _get_float(t, "l", 0.0)
        first = _get_int(t, "first_frame", 0)

        poses_node = t.find("poses")
        poses: List[BoxPose] = []
        if poses_node is not None:
            # poses/item がフレーム列
            for k, p in enumerate(poses_node.findall("item")):
                tx = _get_float(p, "tx", 0.0)
                ty = _get_float(p, "ty", 0.0)
                tz = _get_float(p, "tz", 0.0)
                rx = _get_float(p, "rx", 0.0)
                ry = _get_float(p, "ry", 0.0)
                rz = _get_float(p, "rz", 0.0)
                poses.append(BoxPose(frame=first + k, tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz))

        out.append(Tracklet(obj_type=obj_type, h=h, w=w, l=l, first_frame=first, poses=poses))
    return out

def build_frame_index(tracklets: List[Tracklet], classes: Optional[List[str]] = None) -> Dict[int, List[tuple]]:
    """
    frame -> list of (obj_type, l, w, h, tx, ty, tz, rx, ry, rz)
    """
    frame_map: Dict[int, List[tuple]] = {}
    for tr in tracklets:
        if classes is not None and tr.obj_type not in classes:
            continue
        for pose in tr.poses:
            frame_map.setdefault(pose.frame, []).append(
                (tr.obj_type, tr.l, tr.w, tr.h, pose.tx, pose.ty, pose.tz, pose.rx, pose.ry, pose.rz)
            )
    return frame_map
