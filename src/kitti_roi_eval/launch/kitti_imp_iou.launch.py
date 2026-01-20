# kitti_roi_eval/launch/kitti_imp_iou.launch.py
import os
import shutil
import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def _ensure_file(path: str):
    if not os.path.isfile(path):
        raise RuntimeError(f"[launch] parameter file not found: {path}")


def _copy_to_dir(src: str, dst_dir: str) -> str:
    _ensure_file(src)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _setup(context, *args, **kwargs):
    run_root = LaunchConfiguration("run_root").perform(context)
    run_tag = LaunchConfiguration("run_tag").perform(context).strip()

    if run_tag == "":
        run_tag = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = os.path.expanduser(os.path.join(run_root, run_tag))
    os.makedirs(run_dir, exist_ok=True)

    # run_dir 配下に「今回使った params 一式」を保存（再現性用）
    params_dir = os.path.join(run_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    share_eval = get_package_share_directory("kitti_roi_eval")

    # install/share 側の params を参照
    player_src = os.path.join(share_eval, "params", "kitti_player.yaml")
    eval_src   = os.path.join(share_eval, "params", "roi_eval_iou.yaml")
    est_src    = os.path.join(share_eval, "params", "important_roi_estimator_kitti.yaml")

    # run_dir/params へコピー（このコピーを Node に渡す）
    player_yaml = _copy_to_dir(player_src, params_dir)
    eval_yaml   = _copy_to_dir(eval_src, params_dir)
    est_yaml    = _copy_to_dir(est_src, params_dir)

    # launch が dict で上書きする値（＝「実行時の実際の値」）
    # ノード側 dump は使わないので、dump_params_* は常に False に落とす
    est_overrides = {
        "dump_params_enable": False,
        "dump_params_dir": run_dir,  # 記録として残す（ノード側では使わない想定）
        "csv_path": os.path.join(run_dir, "roi_est_proc_time.csv"),
    }
    eval_overrides = {
        "out_dir": run_dir,
        "dump_params_enable": False,
        "dump_params_dir": run_dir,
    }
    player_overrides = {
        "dump_params_enable": False,
        "dump_params_dir": run_dir,
    }

    # 上書き分も run_dir に保存（「YAMLコピー」＋「上書き記録」で再現可能にする）
    _write_text(
        os.path.join(run_dir, "RUN_INFO.txt"),
        (
            f"run_dir: {run_dir}\n"
            f"created: {datetime.datetime.now().isoformat()}\n"
            f"player_yaml: {player_yaml}\n"
            f"est_yaml: {est_yaml}\n"
            f"eval_yaml: {eval_yaml}\n"
            "\n"
            "note:\n"
            "  - parameter YAMLs are copied into run_dir/params.\n"
            "  - launch_overrides.yaml records dict overrides passed from this launch.\n"
        ),
    )

    # 手書きYAML（PyYAML不要）で上書き内容を記録
    launch_overrides_path = os.path.join(run_dir, "launch_overrides.yaml")
    _write_text(
        launch_overrides_path,
        (
            "kitti_player_with_gt:\n"
            "  ros__parameters:\n"
            f"    dump_params_enable: false\n"
            f"    dump_params_dir: \"{run_dir}\"\n"
            "\n"
            "important_roi_estimator:\n"
            "  ros__parameters:\n"
            f"    dump_params_enable: false\n"
            f"    dump_params_dir: \"{run_dir}\"\n"
            f"    csv_path: \"{os.path.join(run_dir, 'roi_est_proc_time.csv')}\"\n"
            "\n"
            "roi_eval_iou:\n"
            "  ros__parameters:\n"
            f"    out_dir: \"{run_dir}\"\n"
            f"    dump_params_enable: false\n"
            f"    dump_params_dir: \"{run_dir}\"\n"
        ),
    )

    return [
        Node(
            package="kitti_roi_eval",
            executable="kitti_player_with_gt",
            name="kitti_player_with_gt",
            output="screen",
            parameters=[
                player_yaml,      # run_dir/params のコピーを使う
                player_overrides, # 上書き
            ],
        ),
        Node(
            package="lidar_roi_nodes",
            executable="important_roi_estimator",
            name="important_roi_estimator",
            output="screen",
            parameters=[
                est_yaml,         # run_dir/params のコピーを使う
                est_overrides,    # 上書き（csv_path 等）
            ],
        ),
        Node(
            package="kitti_roi_eval",
            executable="roi_eval_iou",
            name="roi_eval_iou",
            output="screen",
            parameters=[
                eval_yaml,        # run_dir/params のコピーを使う
                eval_overrides,   # 上書き（out_dir）
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "run_root",
                default_value=os.path.expanduser("~/ws_livox/result/kitti_imp_iou"),
            ),
            DeclareLaunchArgument(
                "run_tag",
                default_value="",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
