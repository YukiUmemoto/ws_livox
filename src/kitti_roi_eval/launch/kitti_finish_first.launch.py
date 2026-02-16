# kitti_roi_eval/launch/kitti_finish_first.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ---- Launch args ----
    drive_dir = LaunchConfiguration("drive_dir")
    start_idx = LaunchConfiguration("start_idx")
    end_idx = LaunchConfiguration("end_idx")
    stride = LaunchConfiguration("stride")
    out_dir = LaunchConfiguration("out_dir")

    # bins
    V = LaunchConfiguration("V")
    H = LaunchConfiguration("H")
    horizontal_fov_deg = LaunchConfiguration("horizontal_fov_deg")
    vertical_fov_deg = LaunchConfiguration("vertical_fov_deg")
    vertical_fov_up_deg = LaunchConfiguration("vertical_fov_up_deg")
    vertical_fov_down_deg = LaunchConfiguration("vertical_fov_down_deg")

    # GT / cover
    gt_npz = LaunchConfiguration("gt_npz")
    cover_tol_bins = LaunchConfiguration("cover_tol_bins")

    # QoS (finish-first 用)
    qos_depth = LaunchConfiguration("qos_depth")
    qos_points_reliable = LaunchConfiguration("qos_points_reliable")
    qos_ack_reliable = LaunchConfiguration("qos_ack_reliable")

    # QoS (roi_cover_logger 用)
    qos_sub_reliable = LaunchConfiguration("qos_sub_reliable")     # pred/omega
    qos_meta_reliable = LaunchConfiguration("qos_meta_reliable")   # frame_idx/points_count
    qos_sub_depth = LaunchConfiguration("qos_sub_depth")
    qos_meta_depth = LaunchConfiguration("qos_meta_depth")
    qos_ack_depth = LaunchConfiguration("qos_ack_depth")

    # topics (必要なら launch 引数で差し替え可能にする)
    points_topic = LaunchConfiguration("points_topic")
    frame_idx_topic = LaunchConfiguration("frame_idx_topic")
    points_count_topic = LaunchConfiguration("points_count_topic")
    ack_topic = LaunchConfiguration("ack_topic")
    pred_topic = LaunchConfiguration("pred_topic")
    omega_topic = LaunchConfiguration("omega_topic")

    csv_name = LaunchConfiguration("csv_name")
    roi_top_percent = LaunchConfiguration("roi_top_percent")

    # stats (Chap7)
    stats_enable = LaunchConfiguration("stats_enable")
    stats_csv_name = LaunchConfiguration("stats_csv_name")
    stats_flush_every = LaunchConfiguration("stats_flush_every")
    stats_write_header = LaunchConfiguration("stats_write_header")
    publish_rel_low = LaunchConfiguration("publish_rel_low")

    return LaunchDescription([
        # ---- Required ----
        DeclareLaunchArgument("drive_dir", default_value=""),
        DeclareLaunchArgument("start_idx", default_value="0"),
        DeclareLaunchArgument("end_idx", default_value="232"),
        DeclareLaunchArgument("stride", default_value="1"),
        DeclareLaunchArgument("V", default_value="128"),
        DeclareLaunchArgument("H", default_value="128"),
        DeclareLaunchArgument("horizontal_fov_deg", default_value="360.0"),
        DeclareLaunchArgument("vertical_fov_deg", default_value="26.8"),
        DeclareLaunchArgument("vertical_fov_up_deg", default_value="2.0"),
        DeclareLaunchArgument("vertical_fov_down_deg", default_value="24.8"),
        DeclareLaunchArgument("out_dir", default_value=""),

        # ---- GT / cover ----
        DeclareLaunchArgument("gt_npz", default_value=""),
        DeclareLaunchArgument("cover_tol_bins", default_value="1"),
        DeclareLaunchArgument("csv_name", default_value="cover_per_frame.csv"),
        DeclareLaunchArgument("roi_top_percent", default_value="10.0"),

        # ---- Stats (Chap7) ----
        DeclareLaunchArgument("stats_enable", default_value="false"),
        DeclareLaunchArgument("stats_csv_name", default_value="roi_stats.csv"),
        DeclareLaunchArgument("stats_flush_every", default_value="30"),
        DeclareLaunchArgument("stats_write_header", default_value="true"),
        DeclareLaunchArgument("publish_rel_low", default_value="false"),

        # ---- QoS (player) ----
        DeclareLaunchArgument("qos_depth", default_value="50"),
        DeclareLaunchArgument("qos_points_reliable", default_value="false"),
        DeclareLaunchArgument("qos_ack_reliable", default_value="true"),

        # ---- QoS (cover logger) ----
        # pred/omega は important_roi_estimator が BEST_EFFORT なので false 推奨
        DeclareLaunchArgument("qos_sub_reliable", default_value="false"),
        # frame_idx/points_count は kitti_player_finish_first が BEST_EFFORT なので false 推奨
        DeclareLaunchArgument("qos_meta_reliable", default_value="false"),
        DeclareLaunchArgument("qos_sub_depth", default_value="50"),
        DeclareLaunchArgument("qos_meta_depth", default_value="50"),
        DeclareLaunchArgument("qos_ack_depth", default_value="50"),

        # ---- Topics ----
        DeclareLaunchArgument("points_topic", default_value="/livox/lidar_perturbed"),
        DeclareLaunchArgument("frame_idx_topic", default_value="kitti_player/frame_idx"),
        DeclareLaunchArgument("points_count_topic", default_value="kitti_player/points_count"),
        DeclareLaunchArgument("ack_topic", default_value="kitti_player/ack_frame_idx"),
        DeclareLaunchArgument("pred_topic", default_value="roi_est/roi_imp_mono8"),
        DeclareLaunchArgument("omega_topic", default_value="roi_est/omega_mono8"),

        # =========================
        # 1) KITTI player (finish-first)
        # =========================
        Node(
            package="kitti_roi_eval",
            executable="kitti_player_finish_first",
            name="kitti_player_finish_first",
            output="screen",
            parameters=[{
                "drive_dir": ParameterValue(drive_dir, value_type=str),
                "start_idx": ParameterValue(start_idx, value_type=int),
                "end_idx": ParameterValue(end_idx, value_type=int),
                "stride": ParameterValue(stride, value_type=int),

                # topics
                "points_topic": ParameterValue(points_topic, value_type=str),
                "frame_idx_topic": ParameterValue(frame_idx_topic, value_type=str),
                "points_count_topic": ParameterValue(points_count_topic, value_type=str),
                "ack_topic": ParameterValue(ack_topic, value_type=str),

                # QoS
                "qos_depth": ParameterValue(qos_depth, value_type=int),
                "qos_points_reliable": ParameterValue(qos_points_reliable, value_type=bool),
                "qos_ack_reliable": ParameterValue(qos_ack_reliable, value_type=bool),
            }],
        ),

        # =========================
        # 2) important_roi_estimator
        #    ※ package/executable/param 名は実装に合わせて必要に応じて調整してください
        # =========================
        Node(
            package="lidar_roi_nodes",               # 実際の package 名に合わせてください
            executable="important_roi_estimator",    # 実際の executable 名に合わせてください
            name="important_roi_estimator",
            output="screen",
            parameters=[{
                # 入力
                "input_topic": ParameterValue(points_topic, value_type=str),

                # bin 設定（複数名を同時に渡して、実装側の実パラメータ名に吸わせる）
                "V": ParameterValue(V, value_type=int),
                "H": ParameterValue(H, value_type=int),
                "num_vertical_bins": ParameterValue(V, value_type=int),
                "num_horizontal_bins": ParameterValue(H, value_type=int),
                "horizontal_fov_deg": ParameterValue(horizontal_fov_deg, value_type=float),
                "vertical_fov_deg": ParameterValue(vertical_fov_deg, value_type=float),
                "vertical_fov_up_deg": ParameterValue(vertical_fov_up_deg, value_type=float),
                "vertical_fov_down_deg": ParameterValue(vertical_fov_down_deg, value_type=float),
                "azimuth_0_to_hfov": True,
                "roi_top_percent": ParameterValue(roi_top_percent, value_type=float),

                # 重い処理はOFF（完走優先）
                "csv_enable": False,
                "viz_enable": False,

                # Chap7 stats CSV
                "stats_enable": ParameterValue(stats_enable, value_type=bool),
                "stats_csv_path": ParameterValue(
                    PathJoinSubstitution([out_dir, stats_csv_name]),
                    value_type=str,
                ),
                "stats_flush_every": ParameterValue(stats_flush_every, value_type=int),
                "stats_write_header": ParameterValue(stats_write_header, value_type=bool),

                # Reliability-only mask
                "publish_rel_low": ParameterValue(publish_rel_low, value_type=bool),
            }],
        ),

        # =========================
        # 3) roi_cover_logger (GT cover evaluation + ACK)
        # =========================
        Node(
            package="kitti_roi_eval",
            executable="roi_cover_logger",
            name="roi_cover_logger",
            output="screen",
            parameters=[{
                # topics
                "pred_topic": ParameterValue(pred_topic, value_type=str),
                "omega_topic": ParameterValue(omega_topic, value_type=str),
                "frame_idx_topic": ParameterValue(frame_idx_topic, value_type=str),
                "points_count_topic": ParameterValue(points_count_topic, value_type=str),
                "ack_topic": ParameterValue(ack_topic, value_type=str),

                # GT / cover
                "gt_npz": ParameterValue(gt_npz, value_type=str),
                "cover_tol_bins": ParameterValue(cover_tol_bins, value_type=int),

                # IO
                "out_dir": ParameterValue(out_dir, value_type=str),
                "csv_name": ParameterValue(csv_name, value_type=str),

                # QoS（roi_cover_logger の実装側 param 名に合わせている）
                "qos_sub_depth": ParameterValue(qos_sub_depth, value_type=int),
                "qos_meta_depth": ParameterValue(qos_meta_depth, value_type=int),
                "qos_ack_depth": ParameterValue(qos_ack_depth, value_type=int),
                "qos_sub_reliable": ParameterValue(qos_sub_reliable, value_type=bool),
                "qos_meta_reliable": ParameterValue(qos_meta_reliable, value_type=bool),
                "qos_ack_reliable": ParameterValue(qos_ack_reliable, value_type=bool),
            }],
        ),
    ])
