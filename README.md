# ws_livox

Livox/LiDAR 点群から **重要 ROI (Region of Interest)** を推定し、信頼度と評価指標を算出する ROS 2 ワークスペースです。  
点群の欠損・攪乱を模擬する **pointcloud_perturber**、ROI 推定を行う **important_roi_estimator**、エッジ・重要度を可視化する **lidar_edge_estimator**、KITTI raw を用いた評価ツール群 **kitti_roi_eval** を含みます。

> 注意: このリポジトリは ROS 2 + Linux 環境を前提としています（`eval/*.sh` は Bash、ROS 2 Humble 想定）。Windows 上では WSL など Linux 環境での実行が必要です。

---

## 1. ディレクトリ構成

- `src/lidar_roi_nodes/`  
  重要 ROI 推定ノードと点群攪乱ノード（rclpy）
- `src/lidar_edge_estimator/`  
  角度ビン上でのエッジ/重要度推定（rclpy）
- `src/kitti_roi_eval/`  
  KITTI raw を再生・評価するノードやオフライン GT 生成
- `eval/`  
  バッチ評価・可視化スクリプト（bash / python）
- `result/`  
  評価結果の出力先（実行時に生成）

---

## 2. 主要パッケージ概要

### 2.1 `lidar_roi_nodes`

#### 2.1.1 `pointcloud_perturber`
**目的**: 点群の欠損 (drop) やスプーフィング (range bias) を模擬し、GT マスクも生成。  

**入力 / 出力**
- 入力: `input_topic`（PointCloud2、既定 `/livox/lidar`）
- 出力:
  - `output_topic`（PointCloud2、既定 `/livox/lidar_perturbed`）
  - `pc_perturber/gt_mask_mono8`（mono8、影響を受けたビン）
  - `pc_perturber/drop_ratio`（Float32）
  - `pc_perturber/bias_m`（Float32）

**主な挙動**
- ビン分割（H/V、FOV）で点群を割り当て
- `drop_mode=random` はパケット単位でランダムにドロップ  
  `drop_mode=burst` は連続バーストドロップを模擬
- `enable_spoof` 有効時は指定 bin 矩形に `spoof_bias_m` を加算（実質距離をスケール）

**代表パラメータ**
- `drop_prob_q`, `packet_points`, `burst_len_B`, `burst_start_prob`
- `spoof_*`（スプーフィング領域）
- `horizontal_fov_deg`, `vertical_fov_deg`, `num_horizontal_bins`, `num_vertical_bins`

#### 2.1.2 `important_roi_estimator`
**目的**: 点群を角度ビンへ集約し、**重要度 (Importance)** と **信頼度 (Reliability)** を推定して ROI を抽出。  

**入力 / 出力**
- 入力: `input_topic`（PointCloud2）
  - 任意で GT/メタ: `gt_mask_topic`, `drop_ratio_topic`, `bias_topic`
- 出力:
  - `roi_est/importance_map`（Float32MultiArray）
  - `roi_est/rel_map`（Float32MultiArray）
  - `roi_est/roi_imp_mono8`（mono8）
  - `roi_est/roi_use_mono8`（mono8）
  - `roi_est/roi_alert_mono8`（mono8）
  - `roi_est/rel_low_mono8`（mono8, `publish_rel_low=true` 時）
  - `roi_est/omega_mono8`（mono8、観測/期待の有効領域）
  - `roi_est/frame_rel`, `roi_est/frame_rel_obs`（Float32）
  - `roi_est/alert_ratio`, `roi_est/alert_ratio_omega`（Float32）

**推定の流れ（要約）**
1. 点群を (V,H) の角度ビンに集約 → range map `R` とヒット数 `N`
2. EMA により期待レンジ `expected_range` と期待ヒット数 `expected_hits` を更新
3. **Importance**  
   - 空間差分（左右/上下のレンジ差）  
   - 時間差分（`R` と `expected_range` の差）  
   - `I = w_s * S + w_t * T`
4. **Reliability**  
   - 未観測 `d_m`、時間差分 `d_d`、ヒット不足 `d_n` を重み付け  
   - `Rel = sigmoid(beta * (c - Q))`
5. `roi_imp`, `roi_use`, `roi_alert` を `tau_rel` で分割

**CSV ログ**
`csv_enable=true` で `roi_est_proc_time.csv` を出力  
列: `frame_idx, stamp_sec, stamp_nanosec, n_points, proc_time_ms`

**Chap7 用 Stats CSV**
`stats_enable=true` で `roi_est_stats.csv` を出力（`stats_csv_path` で変更可）  
列: `frame_idx, omega_bins, roi_bins, keepbin_ratio, roi_points, keeppt_ratio, S_mean_roi, T_mean_roi, I_mean_roi, frame_rel_all, frame_rel_obs, alert_ratio, alert_ratio_omega, drop_ratio, bias_m`

**注意**  
`src/lidar_roi_nodes/config/roi_est_case*.yaml` は旧パラメータ名が混在しています  
（例: `w_r`, `dr_scale_m`, `publish_maps_as_image` など）。  
現行ノードで有効なパラメータは `important_roi_estimator.py` の `declare_parameter()` を参照してください。

---

### 2.2 `lidar_edge_estimator`

#### `angle_bin_edge_estimator`
**目的**: 角度ビン上の **空間/時間エッジ** と **重要度マップ** を算出し可視化。  

**主な出力**
- `edge_estimator_ema/range_map`
- `edge_estimator_ema/temporal_edge_map`
- `edge_estimator_ema/spatial_edge_map`
- `edge_estimator_ema/importance_map`
加えて 32FC1 と mono8 画像 (`*_img32`, `*_mono8`) を publish。

**可視化**
- `bin_pyramids` Marker を出力（角度ビンの枠線）
- `bin_pyramids_reset_service` で再描画可能

**Launch**
- `ros2 launch lidar_edge_estimator angle_bin_edge_estimator.launch.py`

---

### 2.3 `kitti_roi_eval`

#### 主要ノード

- `kitti_player_with_gt`  
  KITTI raw の velodyne を再生し `PointCloud2` + `frame_idx` を publish。  
  `use_sim_stamp` を使うと frame_idx から疑似 timestamp を生成。

- `kitti_player_finish_first`  
  finish-first 方式のプレイヤー。  
  `ack_frame_idx` を受け取るまで同一フレームを再送し続ける。

- `roi_eval_iou`  
  `pred/gt/omega` をタイムスタンプで同期し IoU/Precision/Recall/F1 を算出、CSV と可視化を保存。

- `roi_cover_logger`  
  オフラインで生成した GT npz を読み込み、finish-first 方式で cover 評価 + ACK。

- `roi_frame_logger`  
  frame_idx を基準に ROI マップ/マスク/GT を保存する軽量ロガー（Chap7 可視化用）。

- `roi_finish_logger` / `roi_min_logger`  
  最小限のログや遅延計測、done 時に CSV を確実に書き出す。

#### GT 生成ユーティリティ
- `gen_gt_binmask.py`  
  tracklet から角度ビンの GT マスク (npz) を生成。
- `offline_generate_gt_bbox.py`  
  BBOX 情報と GT マスクをオフラインで保存。

---

## 3. 依存関係

### 必須
- ROS 2 Humble（rclpy）
- Python 3
- `numpy`

### 評価・可視化で使用
- `matplotlib`（`roi_eval_iou.py`）
- `pandas`（`eval/plot_drop_sweep.py`）
- `imageio`（`offline_generate_gt_bbox.py` の PNG 保存時）

例:
```bash
pip install numpy matplotlib pandas imageio
```
環境によっては `pip` が無い場合があるため、その際は:
```bash
python3 -m pip install imageio
```

---

## 4. ビルド

```bash
cd ~/ws_livox
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 5. 実行手順（代表例）

### 5.1 Livox / ROS bag → 重要 ROI 推定

1) 点群攪乱
```bash
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=0.10
```

2) ROI 推定
```bash
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p csv_enable:=true -p csv_path:=/tmp/roi_est_proc_time.csv
```

3) bag 再生
```bash
ros2 bag play /path/to/bag --rate 1.0 --disable-keyboard-controls
```

---

### 5.2 エッジ推定（可視化）

```bash
ros2 launch lidar_edge_estimator angle_bin_edge_estimator.launch.py
```

`src/lidar_edge_estimator/config/angle_bin_edge_estimator.yaml` で各種閾値・FOV・bin 数を調整。

---

### 5.3 KITTI 評価（IoU）

1) `src/kitti_roi_eval/params/kitti_player.yaml` の `drive_dir` を実データへ変更  
2) 実行:
```bash
ros2 launch kitti_roi_eval kitti_imp_iou.launch.py \
  run_root:=~/ws_livox/result/kitti_imp_iou \
  run_tag:=run_$(date +%Y%m%d_%H%M%S)
```

出力: `run_root/run_tag/` に CSV / 画像 / まとめファイルが生成されます。

---

### 5.4 KITTI finish-first 評価（cover）

1) GT npz を生成
```bash
python3 -m kitti_roi_eval.gen_gt_binmask \
  --drive_dir /path/to/2011_09_26_drive_xxxx_sync \
  --out_npz /path/to/gt_masks.npz \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8
```

固定パス例（V/Hごとにフォルダ分けして保存）:
```bash
mkdir -p /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128
python3 -m kitti_roi_eval.gen_gt_binmask \
  --drive_dir /home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  --out_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8

mkdir -p /home/agx-orin-07/ws_livox/data/gt_masks/V128_H256
python3 -m kitti_roi_eval.gen_gt_binmask \
  --drive_dir /home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  --out_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H256/gt_binmask_V128_H256.npz \
  --V 128 --H 256 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8
```

2) finish-first launch
```bash
ros2 launch kitti_roi_eval kitti_finish_first.launch.py \
  drive_dir:=/path/to/2011_09_26_drive_xxxx_sync \
  gt_npz:=/path/to/gt_masks.npz \
  out_dir:=/path/to/out_cover
```

---

## 6. パラメータファイル

主要なパラメータ例:

- `src/lidar_roi_nodes/config/pc_perturber_random_q0.10.yaml`
- `src/lidar_roi_nodes/config/roi_est_case0_baseline.yaml`
- `src/lidar_edge_estimator/config/angle_bin_edge_estimator.yaml`
- `src/kitti_roi_eval/params/*.yaml`

注意:
`roi_est_case*.yaml` は古いパラメータ名が含まれているため、  
**現行の `important_roi_estimator` に合わせる場合は修正が必要**です。

---

## 7. 評価スクリプト

### 7.1 ドロップ率スイープ
`eval/run_drop_sweep.sh` は複数の drop_prob_q を回して `frames.csv` を生成します。  
出力を `eval/plot_drop_sweep.py` で集計・グラフ化できます。

```bash
export OUT=~/ws_livox/eval/drop_sweep
python3 eval/plot_drop_sweep.py
```

### 7.2 単一 q のスナップショット
`eval/snapshot_one_q.sh` は特定フレーム周辺の画像を保存します。

---

## 7.3 第7章（性能評価）向け 自動評価ツール

第7章で必要な **keepbin/τR スイープ・時系列・重畳可視化・統合評価・処理時間分布** を
同じ手順で再実行できるよう、`eval/chap7/` にスクリプト群を追加しています。

### 7.3.1 keepbin（τ_I 相当）スイープ → 図7.1〜7.3
1) スイープ実行:
```bash
export DRIVE_DIR=/path/to/kitti_drive
export GT_NPZ=/path/to/gt_masks.npz
export OUT=~/ws_livox/eval/chap7/keepbin_sweep
export V=128
export H=128
# rho sweep (%): default now reaches 50% (= rho 0.50)
export P_LIST="1 2 5 10 15 20 30 40 50"
bash eval/chap7/run_keepbin_sweep.sh
```
2) 図生成:
```bash
python3 eval/chap7/plot_keepbin_sweep.py --root ~/ws_livox/eval/chap7/keepbin_sweep
```
出力:  
`fig_bbox_cov_vs_keepbin_tauI.png` / `fig_bbox_cov_random_baseline_keepbin.png` / `fig_keeppt_vs_keepbin_tauI.png`

### 7.3.2 代表フレーム抽出 & 重畳可視化 → 図7.6
1) 代表フレーム抽出:
```bash
python3 eval/chap7/select_representative_frames.py \\
  --cover_csv /path/to/cover_per_frame.csv \\
  --gt_npz /path/to/gt_masks.npz \\
  --out_json /path/to/rep_frames_VxH.json
```
2) 重要度マップ/ROI/GT の保存（`important_roi_estimator` と KITTI player を動かした状態で `roi_frame_logger` を起動）:
```bash
ros2 run kitti_roi_eval roi_frame_logger --ros-args \\
  -p out_dir:=/path/to/frames_out/VxH \\
  -p save_importance:=true -p save_roi_masks:=true \\
  -p save_format:=png -p save_npy:=false \\
  -p save_gt_mask:=true \\
  -p gt_npz_path:=/path/to/gt_masks.npz \\
  -p split_masks_by_type:=true \\
  -p save_vis_aligned:=true \\
  -p vis_hfov_deg:=360.0 -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312 \
  -p vis_center_az:=true
```
3) 重畳レンダリング（`vis/` があれば自動で使用）:
```bash
python3 eval/chap7/render_kitti_overlay.py \\
  --maps_dir /path/to/frames_out/VxH \\
  --gt_npz /path/to/gt_masks.npz \\
  --frames_json /path/to/rep_frames_VxH.json \\
  --out_dir /path/to/figs/VxH
```

3-b) 全フレーム重畳を一括保存（代表3枚ではなく全保存）:
```bash
python3 eval/chap7/render_kitti_overlay.py \\
  --maps_dir /path/to/frames_out/VxH \\
  --gt_npz /path/to/gt_masks.npz \\
  --all_frames \\
  --all_prefix overlay_ \\
  --out_dir /path/to/figs_all/VxH
```

補足: `roi_frame_logger` は通常の `maps/`・`masks/` に加えて、**可視化用に整形した `vis/` を出力**できます。  
`vis/` には以下の処理が適用された画像/マスクが入ります（評価値は変えず見た目だけ調整）:
- 上下反転・左右反転（画像座標に合わせるため）
- 方位角0°が中央になるよう水平ロール（左右分裂の解消）
- 縦横比を hfov/vfov に合わせて横長化

可視化系スクリプト（`render_*_overlay.py`）は **`vis/` が存在すれば自動で優先**します。  
論文図には `vis/` 出力を使うことを推奨します。

### 7.3.2a GT妥当性チェック（GT形状 + GT内点数）
GTマスクの形状（連結成分数・bbox範囲）と、GTビン内の点数をCSV出力:
```bash
python3 eval/chap7/gt_quality_check.py \
  --drive_dir /path/to/kitti_drive \
  --gt_npz /path/to/gt_masks.npz \
  --out_csv /path/to/gt_quality/VxH/gt_quality.csv \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8
```

### 7.3.2b 画像上へのGT重畳（視覚チェック）
KITTI画像に、GTビン集合に該当する点群を投影して重畳:
```bash
python3 eval/chap7/render_gt_bins_on_image.py \
  --drive_dir /path/to/kitti_drive \
  --gt_npz /path/to/gt_masks.npz \
  --out_dir /path/to/gt_overlay_images/VxH \
  --cam 2 \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8
```

### 7.3.2c 画像上へのGTビン境界線（底面）重畳
GTビン集合の **ビン境界（底面の線分）** を画像に投影して重畳:
```bash
python3 eval/chap7/render_gt_bin_edges_on_image.py \
  --drive_dir /path/to/kitti_drive \
  --gt_npz /path/to/gt_masks.npz \
  --out_dir /path/to/gt_bin_edges/VxH \
  --cam 2 \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8 \
  --r_line 20
```

### 7.3.3 τ_R スイープ（欠損形状比較）→ 図7.7
```bash
export BAG=/path/to/bag
export OUT=~/ws_livox/eval/chap7/tauR_sweep
export P=0.10
bash eval/chap7/run_tauR_sweep.sh
python3 eval/chap7/plot_tauR_sweep.py --root ~/ws_livox/eval/chap7/tauR_sweep/p_0.10
```
出力: `fig_cov_miss_vs_tauR.png`

### 7.3.4 重要度推移（S/T/I 平均）→ 図7.4/7.5
```bash
python3 eval/chap7/plot_importance_timeseries.py \\
  --csvs /path/to/static/roi_stats.csv /path/to/dynamic/roi_stats.csv \\
  --labels static dynamic \\
  --out /path/to/fig_imp_static_vs_dynamic.png
```
KITTI 系列も同様に `--csvs` を1つ指定して出力します。

### 7.3.5 信頼度推移（p 複数値）→ 図7.8/7.9
`roi_est_stats.csv` の `frame_rel_all` を時系列で描画:
```bash
python3 eval/chap7/plot_reliability_timeseries.py \\
  --root /path/to/q_sweep_root \\
  --metric frame_rel_all \\
  --out /path/to/fig_rel_timeseries.png
```

### 7.3.6 欠損真値×低信頼の重畳 → 図7.10
`roi_frame_logger` を使って欠損GTと低信頼マスクを保存:
```bash
ros2 run kitti_roi_eval roi_frame_logger --ros-args \\
  -p out_dir:=/path/to/frames_out \\
  -p save_gt_mask:=true -p save_rel_low:=true -p save_roi_masks:=true \\
  -p save_format:=png -p save_npy:=false \\
  -p split_masks_by_type:=true \\
  -p save_vis_aligned:=true \\
  -p vis_hfov_deg:=360.0 -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312 \
  -p vis_center_az:=true
```
```bash
python3 eval/chap7/render_missing_overlay.py \\
  --maps_dir /path/to/frames_out \\
  --frame_idx 200 \\
  --use_rel_low \\
  --out /path/to/fig_miss_vis.png
```

### 7.3.7 統合評価（重要度/信頼度/統合の比較）→ 図7.12
まず統合評価の3条件を取得:
```bash
export BAG=/path/to/bag
export OUT=~/ws_livox/eval/chap7/integration_eval
bash eval/chap7/run_integration_eval.sh
```
統合出力の看板図（図7.11）:
```bash
ros2 run kitti_roi_eval roi_frame_logger --ros-args \\
  -p out_dir:=/path/to/frames_out \\
  -p save_roi_masks:=true -p save_rel_low:=true \\
  -p save_format:=png -p save_npy:=false \\
  -p split_masks_by_type:=true \\
  -p save_vis_aligned:=true \\
  -p vis_hfov_deg:=360.0 -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312 \
  -p vis_center_az:=true
```
```bash
python3 eval/chap7/render_integration_overlay.py \\
  --maps_dir /path/to/frames_out \\
  --frame_idx 200 \\
  --out /path/to/fig_integration_vis_kitti_rep.png
```
`roi_eval_iou` を3条件で回して `iou_per_frame.csv` を得た後:
```bash
python3 eval/chap7/plot_integration_metrics.py \\
  --imp_csv /path/to/imp/iou_per_frame.csv \\
  --rel_csv /path/to/rel/iou_per_frame.csv \\
  --int_csv /path/to/int/iou_per_frame.csv \\
  --out /path/to/fig_cov_integrated.png
```

### 7.3.8 処理時間分布・追従率 → 図7.13/7.14
```bash
python3 eval/chap7/plot_proc_time.py \\
  --csv /path/to/roi_est_proc_time.csv \\
  --period_ms 100 \\
  --out /path/to/fig_proc_time.png
```

### 7.3.9 Chap7 一括実行テンプレート
```bash
bash eval/chap7/run_chap7_all.sh
```

### 7.3.10 第7章フル実行（手動）
重複を避けるため、**固定パス版の手順は 7.4** に集約しています。  
ローカル環境で一括実行したい場合は、`7.4` の各セクションを上から順に実行してください。

---

## 7.4 この環境での実行プロンプト（固定パス版）

このリポジトリ内に実在するパスへ固定した **コピペ用プロンプト** です。  
（KITTI と bag は `/home/agx-orin-07/ws_livox/` 配下に存在する前提）

### 7.4.0 GT作成（角度ビンGT）
```bash
python3 -m kitti_roi_eval.gen_gt_binmask \
  --drive_dir /home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  --out_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8
```

可視化と同じ向き（例: 左右反転 + 水平ロール）で GT を保存したい場合:
```bash
python3 -m kitti_roi_eval.gen_gt_binmask \
  --drive_dir /home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  --out_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128_visalign.npz \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8 \
  --align_flip_lr --align_center_az
```
`roi_cover_logger` は GT npz 内の `align_*` メタデータを自動読込し、`pred/omega` 側にも同じ整列を適用して比較します（形状・向きを統一）。

注: 評価（Cov/IoU/時系列）を行う場合は、まずは **align オプションなし** の GT を基準にしてください。
可視化だけを vis 向きにしたい場合のみ `--align_flip_lr --align_center_az` 付き GT を別ファイルとして使い分けます。

### 共通（端末ごとに一度）
```bash
source /opt/ros/humble/setup.bash
source /home/agx-orin-07/ws_livox/install/setup.bash
```

### 共通（重要: GTと内部処理の向きを統一）
以下を全評価で統一してください（GT生成時と同じ値）:
```bash
export HORIZONTAL_FOV_DEG=360.0
export VERTICAL_FOV_DEG=26.8
export VERTICAL_FOV_UP_DEG=2.0
export VERTICAL_FOV_DOWN_DEG=24.8
```
`run_keepbin_sweep.sh / run_q_sweep_stats.sh / run_tauR_sweep.sh / run_integration_eval.sh` はこの値を
`pointcloud_perturber` / `important_roi_estimator` に渡し、同じ VxH ビングリッドで比較します。

### 図7.1〜7.3：keepbin スイープ（KITTI）
```bash
export DRIVE_DIR=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
export GT_NPZ=/home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/keepbin_sweep/V128_H128
export V=128
export H=128
export HORIZONTAL_FOV_DEG=360.0
export VERTICAL_FOV_DEG=26.8
export P_LIST="1 2 5 10 15 20 30 40 50"
bash /home/agx-orin-07/ws_livox/eval/chap7/run_keepbin_sweep.sh
python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_keepbin_sweep.py --root /home/agx-orin-07/ws_livox/eval/chap7/keepbin_sweep/V128_H128
```

H=256 版（同様）:
```bash
export DRIVE_DIR=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
export GT_NPZ=/home/agx-orin-07/ws_livox/data/gt_masks/V128_H256/gt_binmask_V128_H256.npz
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/keepbin_sweep/V128_H256
export V=128
export H=256
export HORIZONTAL_FOV_DEG=360.0
export VERTICAL_FOV_DEG=26.8
export P_LIST="1 2 5 10 15 20 30 40 50"
bash /home/agx-orin-07/ws_livox/eval/chap7/run_keepbin_sweep.sh
python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_keepbin_sweep.py --root /home/agx-orin-07/ws_livox/eval/chap7/keepbin_sweep/V128_H256
```

注:
- `run_keepbin_sweep.sh` は `GT_NPZ` の `V/H` と `export V,H` が一致しない場合、実行前にエラーで停止します（`(128,256) vs (128,128)` の混在防止）。
- `gt_empty` が極端に多い場合は FOV 不一致の可能性が高いです。`HORIZONTAL_FOV_DEG` / `VERTICAL_FOV_DEG` を GT 作成時（例: 360/26.8）と一致させてください。

### 図7.4/7.5：重要度推移（S/T/I）
静的/動的 bag から stats CSV を生成して描画:
```bash
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/imp_static
export QS="0.00"
bash /home/agx-orin-07/ws_livox/eval/chap7/run_q_sweep_stats.sh

export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/imp_dynamic
export QS="0.00"
bash /home/agx-orin-07/ws_livox/eval/chap7/run_q_sweep_stats.sh

python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_importance_timeseries.py \
  --csvs \
    /home/agx-orin-07/ws_livox/eval/chap7/imp_static/q_0.00/roi_stats.csv \
    /home/agx-orin-07/ws_livox/eval/chap7/imp_dynamic/q_0.00/roi_stats.csv \
  --labels static dynamic \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_imp_static_vs_dynamic.png
```

run_q_sweep_stats.sh が動かない環境向け（bash の代わりに直接コマンド）:

Terminal A（perturber）:
```bash
source /opt/ros/humble/setup.bash
source /home/agx-orin-07/ws_livox/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/agx-orin-07/ws_livox/eval/chap7/fastrtps_no_shm.xml

ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p sub_reliability:=reliable -p sub_durability:=volatile \
  -p pub_reliability:=best_effort \
  -p enable_drop:=true -p drop_prob_q:=0.00
```

Terminal B（important_roi_estimator）:
```bash
source /opt/ros/humble/setup.bash
source /home/agx-orin-07/ws_livox/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/agx-orin-07/ws_livox/eval/chap7/fastrtps_no_shm.xml

mkdir -p /home/agx-orin-07/ws_livox/eval/chap7/imp_static/q_0.00

ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p roi_top_percent:=10.0 \
  -p stats_enable:=true \
  -p stats_csv_path:=/home/agx-orin-07/ws_livox/eval/chap7/imp_static/q_0.00/roi_stats.csv \
  -p stats_flush_every:=1

ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p roi_top_percent:=10.0 \
  -p stats_enable:=true \
  -p stats_csv_path:=/home/agx-orin-07/ws_livox/eval/chap7/imp_dynamic/q_0.00/roi_stats.csv \
  -p stats_flush_every:=1
```

Terminal C（bag play）:
```bash
source /opt/ros/humble/setup.bash
source /home/agx-orin-07/ws_livox/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/agx-orin-07/ws_livox/eval/chap7/fastrtps_no_shm.xml

ros2 bag play /home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01 \
  --rate 1.0 --disable-keyboard-controls

ros2 bag play /home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01 \
  --rate 1.0 --disable-keyboard-controls
```


KITTI 対応版（方法A: finish-first + ACK を使って stats CSV を生成）:
```bash
# p（drop率）ごとに出力フォルダを分ける，適切に変更
export P=0.00
export P=0.10
# KITTI用FOV（必ず統一）
export KITTI_FOV_ARGS="-p horizontal_fov_deg:=360.0 -p vertical_fov_deg:=26.8 -p vertical_fov_up_deg:=2.0 -p vertical_fov_down_deg:=24.8 -p azimuth_0_to_hfov:=true"

# Terminal A: important_roi_estimator
export P=0.10
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p roi_top_percent:=10.0 \
  -p stats_enable:=true \
  -p stats_csv_path:=/home/agx-orin-07/ws_livox/eval/chap7/imp_kitti/p_${P}/roi_stats.csv \
  -p stats_flush_every:=1

# Terminal B: ACK logger（ACK を返さないと finish-first が止まる）
export P=0.10
ros2 run kitti_roi_eval roi_finish_logger --ros-args \
  -p out_dir:=/home/agx-orin-07/ws_livox/eval/chap7/imp_kitti/p_${P} \
  -p csv_name:=min_log.csv \
  -p pred_topic:=roi_est/roi_imp_mono8 \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed

# Terminal C: KITTI player（finish-first）
export P=0.10
ros2 run kitti_roi_eval kitti_player_finish_first --ros-args \
  -p drive_dir:=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  -p points_topic:=/livox/lidar_perturbed

python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_importance_timeseries.py \
  --csvs /home/agx-orin-07/ws_livox/eval/chap7/imp_kitti/p_${P}/roi_stats.csv \
  --labels kitti \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_kitti_imp_timeseries.png
```

補足: RTPS_TRANSPORT_SHM エラーが出る環境では、**ターミナルを3つ**に分けて
`pointcloud_perturber` / `important_roi_estimator` / `ros2 bag play` を独立起動すると
安定して CSV が生成できることを確認済みです。

補足: `roi_stats.csv` に `roi_top_percent` を出力するように追加済みです。


### 図7.6：代表フレーム重畳（KITTI）
```bash
# 0) 代表フレーム抽出（Cov_bbox 分布から典型/上位5%/下位5%を選択）
python3 /home/agx-orin-07/ws_livox/eval/chap7/select_representative_frames.py \
  --cover_csv /home/agx-orin-07/ws_livox/eval/chap7/keepbin_sweep/V128_H128/p_10/cover_per_frame.csv \
  --gt_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --out_json /home/agx-orin-07/ws_livox/eval/chap7/rep_frames_V128_H128.json

# 1) KITTI player + important_roi_estimator を起動して map を流す
#   - roi_frame_logger は「入力が来たときだけ」保存するので、必ず同時に起動する
#   - finish-first ではなく通常の player を使う（ACK 不要）

# Terminal A: important_roi_estimator
export KITTI_FOV_ARGS="-p horizontal_fov_deg:=360.0 -p vertical_fov_deg:=26.8 -p vertical_fov_up_deg:=2.0 -p vertical_fov_down_deg:=24.8 -p azimuth_0_to_hfov:=true"

ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p roi_top_percent:=10.0 \
  -p publish_rel_low:=true \
  -p stats_enable:=false

# Terminal B: roi_frame_logger を別ターミナルで起動（保存先は固定）
ros2 run kitti_roi_eval roi_frame_logger --ros-args \
  -p out_dir:=/home/agx-orin-07/ws_livox/eval/chap7/kitti_frames/V128_H128 \
  -p save_importance:=true -p save_roi_masks:=true \
  -p save_format:=png -p save_npy:=false \
  -p save_gt_mask:=true \
  -p gt_npz_path:=/home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  -p split_masks_by_type:=true \
  -p save_vis_aligned:=true \
  -p vis_hfov_deg:=360.0 -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312 \
  -p vis_center_az:=true

# Terminal C: KITTI player（全フレーム配信）
ros2 run kitti_roi_eval kitti_player_with_gt --ros-args \
  -p drive_dir:=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  -p points_topic:=/livox/lidar_perturbed

# 3) 保存された map/mask から重畳図を作成
#    ※ render_kitti_overlay.py は既定で --align_mode auto です。
#      GT(npzメタ情報)と入力ソース(raw/vis)から可視化変換を自動一致させるため、
#      追加のflip/roll指定は不要です（内部評価処理には影響しません）。
python3 /home/agx-orin-07/ws_livox/eval/chap7/render_kitti_overlay.py \
  --maps_dir /home/agx-orin-07/ws_livox/eval/chap7/kitti_frames/V128_H128 \
  --gt_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --frames_json /home/agx-orin-07/ws_livox/eval/chap7/rep_frames_V128_H128.json \
  --out_dir /home/agx-orin-07/ws_livox/eval/chap7/figs_kitti_rep/V128_H128

# 4) 全フレームの重畳も保存したい場合
python3 /home/agx-orin-07/ws_livox/eval/chap7/render_kitti_overlay.py \
  --maps_dir /home/agx-orin-07/ws_livox/eval/chap7/kitti_frames/V128_H128 \
  --gt_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --all_frames \
  --all_prefix overlay_ \
  --out_dir /home/agx-orin-07/ws_livox/eval/chap7/figs_kitti_all/V128_H128

# 備考:
# - 既定では --use_logged_gt が有効で、roi_frame_logger が保存した gt_mask を優先して重畳します。
#   これにより、vis 側の変換（flip/roll/stretch）と完全一致した座標で表示されます。
# - gt_npz から直接重畳したい場合のみ --no_use_logged_gt を付けてください。
```


### 図7.6補足：GT妥当性チェック（GT形状 + GT内点数）
```bash
python3 /home/agx-orin-07/ws_livox/eval/chap7/gt_quality_check.py \
  --drive_dir /home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  --gt_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --out_csv /home/agx-orin-07/ws_livox/eval/chap7/gt_quality/V128_H128/gt_quality.csv \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8
```

### 図7.6補足：画像上へのGT重畳（視覚チェック）
```bash
python3 /home/agx-orin-07/ws_livox/eval/chap7/render_gt_bins_on_image.py \
  --drive_dir /home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  --gt_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --out_dir /home/agx-orin-07/ws_livox/eval/chap7/gt_overlay_images/V128_H128 \
  --cam 2 \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8
```

### 図7.6補足：画像上へのGTビン境界線（底面）重畳
```bash
python3 /home/agx-orin-07/ws_livox/eval/chap7/render_gt_bin_edges_on_image.py \
  --drive_dir /home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync \
  --gt_npz /home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz \
  --out_dir /home/agx-orin-07/ws_livox/eval/chap7/gt_bin_edges/V128_H128 \
  --cam 2 \
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 26.8 \
  --vfov_up_deg 2.0 --vfov_down_deg 24.8 \
  --r_line 20
```




### 図7.7：τ_R スイープ（欠損形状比較）
```bash
# dynamic bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/tauR_sweep/dynamic/p_0.10
export P=0.10
export V=128
export H=128
# Terminal A: perturber
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P
# Terminal B: estimator（tau_rel を変えてスイープ）
export TAU_REL=0.80
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p tau_rel:=$TAU_REL \
  -p publish_rel_low:=true \
  -p stats_enable:=true \
  -p stats_csv_path:=$OUT/random/tauR_${TAU_REL}/roi_stats.csv \
  -p stats_flush_every:=1
# Terminal C: roi_eval_iou（欠損カバー率）
mkdir -p $OUT/random/tauR_${TAU_REL}
ros2 run kitti_roi_eval roi_eval_iou --ros-args \
  -p pred_topic:=roi_est/rel_low_mono8 \
  -p gt_topic:=pc_perturber/gt_mask_mono8 \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed \
  -p out_dir:=$OUT/random/tauR_${TAU_REL} \
  -p csv_name:=iou_per_frame.csv
# Terminal D: bag play
ros2 bag play $BAG --rate 1.0 --disable-keyboard-controls
python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_tauR_sweep.py \
  --root /home/agx-orin-07/ws_livox/eval/chap7/tauR_sweep/dynamic/p_0.10




# static bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/tauR_sweep/static/p_0.05
export V=128
export H=128
export P=0.05
export TAU_REL=0.85

# export REAL_FOV_ARGS="-p horizontal_fov_deg:=70.4 -p vertical_fov_deg:=77.2 -p vertical_fov_up_deg:=-1.0 -p vertical_fov_down_deg:=-1.0 -p azimuth_0_to_hfov:=false"

# Terminal A: perturber
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p azimuth_0_to_hfov:=false \
  -p horizontal_fov_deg:=70.4 \
  -p vertical_fov_deg:=77.2 \
  -p vertical_fov_up_deg:=-1.0 \
  -p vertical_fov_down_deg:=-1.0 \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P
  
# Terminal B: estimator（tau_rel を変えてスイープ）
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p azimuth_0_to_hfov:=false \
  -p horizontal_fov_deg:=70.4 \
  -p vertical_fov_deg:=77.2 \
  -p vertical_fov_up_deg:=-1.0 \
  -p vertical_fov_down_deg:=-1.0 \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p tau_rel:=$TAU_REL \
  -p publish_rel_low:=true \
  -p stats_enable:=true \
  -p stats_csv_path:=$OUT/random/tauR_${TAU_REL}/roi_stats.csv \
  -p stats_flush_every:=1

# Terminal C: roi_eval_iou（欠損カバー率）
mkdir -p $OUT/random/tauR_${TAU_REL}

ros2 run kitti_roi_eval roi_eval_iou --ros-args \
  -p pred_topic:=roi_est/rel_low_mono8 \
  -p gt_topic:=pc_perturber/gt_mask_mono8 \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed \
  -p out_dir:=$OUT/random/tauR_${TAU_REL} \
  -p csv_name:=iou_per_frame.csv \
  -p vis_center_az:=false

# Terminal D: bag play
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01

ros2 bag play $BAG --rate 0.2 --disable-keyboard-controls
ros2 bag play $BAG --rate 1.0 --disable-keyboard-controls


python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_tauR_sweep.py \
  --root /home/agx-orin-07/ws_livox/eval/chap7/tauR_sweep/static/p_0.05







# KITTI（bagなし）
export DRIVE_DIR=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/tauR_sweep/kitti/p_0.10
export RUN_DIR=$OUT/random/tauR_${TAU_REL}
export P=0.10
export TAU_REL=0.85
export V=128
export H=128

export KITTI_FOV_ARGS="-p horizontal_fov_deg:=360.0 -p vertical_fov_deg:=26.8 -p vertical_fov_up_deg:=2.0 -p vertical_fov_down_deg:=24.8 -p azimuth_0_to_hfov:=true"


# Terminal A: perturber
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P

# Terminal B: estimator（tau_rel を変更してスイープ）
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p tau_rel:=$TAU_REL \
  -p publish_rel_low:=true

# Terminal C: roi_eval_iou（plot用CSV）
export RUN_DIR=$OUT/random/tauR_${TAU_REL}

ros2 run kitti_roi_eval roi_eval_iou --ros-args \
  -p pred_topic:=roi_est/rel_low_mono8 \
  -p gt_topic:=pc_perturber/gt_mask_mono8 \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed \
  -p out_dir:=$RUN_DIR \
  -p csv_name:=iou_per_frame.csv \
  -p viz_enable:=true \
  -p viz_max_frames:=100000

# Terminal D: kitti_player_with_gt
ros2 run kitti_roi_eval kitti_player_with_gt --ros-args \
  -p drive_dir:=$DRIVE_DIR \
  -p points_topic:=/livox/lidar

# Terminal E: KITTIフレーム生成（vis）
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/tauR_sweep/kitti/p_0.10
export OUT_FRAMES=$OUT/frames_kitti/tauR_${TAU_REL}

ros2 run kitti_roi_eval roi_frame_logger --ros-args \
  -p out_dir:="$OUT_FRAMES" \
  -p frame_idx_topic:=kitti_player/frame_idx \
  -p save_enable:=true \
  -p save_roi_masks:=true \
  -p save_rel_low:=true \
  -p save_gt_mask:=true \
  -p split_masks_by_type:=true \
  -p save_format:=png \
  -p save_npy:=false \
  -p save_vis_aligned:=true \
  -p vis_flip_ud:=true \
  -p vis_flip_lr:=true \
  -p vis_center_az:=true \
  -p vis_stretch:=true \
  -p vis_hfov_deg:=360.0 \
  -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312





python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_tauR_sweep.py \
  --root $OUT --skip_frames 50
```


補足: `run_tauR_sweep.sh` を使う場合は内部で以下を起動します（別途起動不要）  
- `pointcloud_perturber`（`pc_perturber/gt_mask_mono8` を生成）  
- `important_roi_estimator`（`roi_est/rel_low_mono8` を生成）  
- `roi_eval_iou`（`iou_per_frame.csv` を出力）  
- `ros2 bag play`  
CSVが空のときは、`$OUT/*/tauR_*/roi_eval.log` と `$OUT/*/tauR_*/estimator.log` を確認してください。





### 図7.8/7.9：信頼度推移（p 複数値）
```bash
# dynamic bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/q_sweep_stats/dynamic
export V=128
export H=128
# Terminal A: perturber
export P=0.10
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P
# Terminal B: estimator（stats CSV）
mkdir -p $OUT/p_${P}
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p stats_enable:=true \
  -p stats_csv_path:=$OUT/p_${P}/roi_stats.csv \
  -p stats_flush_every:=1
# Terminal C: bag play
ros2 bag play $BAG --rate 1.0 --disable-keyboard-controls

python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_reliability_timeseries.py \
  --root /home/agx-orin-07/ws_livox/eval/chap7/q_sweep_stats/dynamic \
  --metric frame_rel_all \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_rel_timeseries_dynamic.png





# static bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/q_sweep_stats/static
export V=128
export H=128
export P=0.00

# Terminal A: perturber
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p horizontal_fov_deg:=70.4 -p vertical_fov_deg:=77.2 \
  -p vertical_fov_up_deg:=-1.0 -p vertical_fov_down_deg:=-1.0 \
  -p azimuth_0_to_hfov:=false \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P \
  -p rng_seed:=0

# Terminal B: estimator（stats CSV, legacy params）
mkdir -p $OUT/p_${P}
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p horizontal_fov_deg:=70.4 -p vertical_fov_deg:=77.2 \
  -p vertical_fov_up_deg:=-1.0 -p vertical_fov_down_deg:=-1.0 \
  -p azimuth_0_to_hfov:=false \
  -p roi_top_percent:=5.0 \
  -p tau_rel:=0.85 \
  -p ema_alpha:=0.05 \
  -p min_expected_hits:=1.0 \
  -p N_min_compare:=1 \
  -p w_s:=0.5 -p w_t:=0.5 \
  -p w_m:=0.45 -p w_n:=0.35 -p w_d:=0.2 \
  -p scale_d_m:=0.50 \
  -p n_floor:=1.0 \
  -p sigmoid_beta:=8.0 -p sigmoid_center_c:=0.55 \
  -p stats_enable:=true \
  -p stats_csv_path:=$OUT/p_${P}/roi_stats.csv \
  -p stats_flush_every:=1

# Terminal C: bag play（先頭5秒除外）
ros2 bag play $BAG --start-offset 5.0 --rate 1.0 --disable-keyboard-controls

python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_reliability_timeseries.py \
  --root /home/agx-orin-07/ws_livox/eval/chap7/q_sweep_stats/static \
  --metric frame_rel_all \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_rel_timeseries_static.png \
  --both





# KITTI（bagなし）
export DRIVE_DIR=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/q_sweep_stats/kitti
export P=0.25
export V=128
export H=128
export KITTI_FOV_ARGS="-p horizontal_fov_deg:=360.0 -p vertical_fov_deg:=26.8 -p vertical_fov_up_deg:=2.0 -p vertical_fov_down_deg:=24.8 -p azimuth_0_to_hfov:=true"

mkdir -p $OUT/p_${P}

# Terminal A: ACK logger
ros2 run kitti_roi_eval roi_finish_logger --ros-args \
  -p out_dir:=$OUT/p_${P} \
  -p csv_name:=min_log.csv \
  -p pred_topic:=roi_est/roi_imp_mono8 \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed

# Terminal B: perturber
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P

# Terminal C: estimator（stats CSV）
# Terminal C: estimator（まず保存確認）
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p roi_top_percent:=5.0 \
  -p tau_rel:=0.85 \
  -p ema_alpha:=0.05 \
  -p min_expected_hits:=1.0 \
  -p N_min_compare:=1 \
  -p w_s:=0.5 -p w_t:=0.5 \
  -p w_m:=0.45 -p w_n:=0.35 -p w_d:=0.2 \
  -p scale_d_m:=0.50 \
  -p n_floor:=1.0 \
  -p sigmoid_beta:=8.0 -p sigmoid_center_c:=0.55 \
  -p stats_enable:=true \
  -p stats_csv_path:="$OUT/p_${P}/roi_stats.csv" \
  -p stats_write_header:=true \
  -p stats_flush_every:=1 \
  -p profile_enable:=false

# Terminal D: kitti_player_finish_first（ACK が必要）
ros2 run kitti_roi_eval kitti_player_finish_first --ros-args \
  -p drive_dir:=$DRIVE_DIR \
  -p points_topic:=/livox/lidar \
  -p republish_interval_sec:=0.2 \
  -p ack_timeout_sec:=30.0 \
  -p qos_points_reliable:=false


# P を変えて繰り返し → timeseries 生成
python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_reliability_timeseries.py \
  --root $OUT \
  --metric frame_rel_all \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_rel_timeseries_kitti.png
```



### 図7.10：欠損真値×低信頼 重畳
```bash
# dynamic bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/miss_frames/dynamic
export P=0.10
export V=128
export H=128

# Terminal A: perturber（欠損付与）
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P

# Terminal B: estimator（低信頼マスク publish）
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p publish_rel_low:=true

# Terminal C: roi_frame_logger（gt_mask + rel_low 保存）
ros2 run kitti_roi_eval roi_frame_logger --ros-args \
  -p out_dir:=$OUT \
  -p save_gt_mask:=true -p save_rel_low:=true -p save_roi_masks:=true \
  -p save_format:=png -p save_npy:=false \
  -p split_masks_by_type:=true \
  -p save_vis_aligned:=true \
  -p vis_hfov_deg:=360.0 -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312 \
  -p vis_center_az:=true

# Terminal D: bag play
ros2 bag play $BAG --rate 1.0 --disable-keyboard-controls

python3 /home/agx-orin-07/ws_livox/eval/chap7/render_missing_overlay.py \
  --maps_dir $OUT \
  --frame_idx 200 \
  --use_rel_low \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_miss_vis_dynamic.png



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ========================================
# static bag
# ========================================
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/miss_frames/static
export P=0.10
export V=128
export H=128
mkdir -p $OUT

# Terminal A: perturber（欠損付与）
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P \
  -p azimuth_0_to_hfov:=false \
  -p horizontal_fov_deg:=70.4 \
  -p vertical_fov_deg:=77.2 \
  -p vertical_fov_up_deg:=-1.0 \
  -p vertical_fov_down_deg:=-1.0

# Terminal B: estimator（低信頼マスク publish）
# Terminal B
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p publish_rel_low:=true \
  -p azimuth_0_to_hfov:=false \
  -p horizontal_fov_deg:=70.4 \
  -p vertical_fov_deg:=77.2 \
  -p vertical_fov_up_deg:=-1.0 \
  -p vertical_fov_down_deg:=-1.0

# Terminal C-1:
python3 - <<'PY'
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32

qos_sub = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

class FrameIdxPub(Node):
    def __init__(self):
        super().__init__('pc_frame_idx_pub')
        self.i = 0
        self.pub = self.create_publisher(Int32, '/kitti_player/frame_idx', 10)
        self.sub = self.create_subscription(PointCloud2, '/livox/lidar_perturbed', self.cb, qos_sub)
    def cb(self, _):
        m = Int32(); m.data = self.i
        self.pub.publish(m)
        self.i += 1

rclpy.init()
n = FrameIdxPub()
rclpy.spin(n)
PY

# Terminal C: roi_frame_logger（gt_mask + rel_low 保存）
ros2 run kitti_roi_eval roi_frame_logger --ros-args \
  -p frame_idx_topic:=/kitti_player/frame_idx \
  -p out_dir:=$OUT \
  -p save_gt_mask:=true -p save_rel_low:=true -p save_roi_masks:=true \
  -p save_format:=png -p save_npy:=false \
  -p split_masks_by_type:=true \
  -p save_vis_aligned:=false \
  -p vis_flip_ud:=false \
  -p vis_flip_lr:=false \
  -p vis_center_az:=false \
  -p vis_stretch:=false

# Terminal D: bag play
ros2 bag play $BAG --rate 1.0 --disable-keyboard-controls

# render（再変換を無効化）
python3 /home/agx-orin-07/ws_livox/eval/chap7/render_missing_overlay.py \
  --maps_dir $OUT \
  --frame_idx 200 \
  --use_rel_low \
  --no_prefer_vis \
  --no_flip_ud \
  --no_flip_lr \
  --no_stretch \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_miss_vis_static.png




# KITTI（bagなし）
export DRIVE_DIR=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/miss_frames/kitti
export P=0.10
export V=128
export H=128
export KITTI_FOV_ARGS="-p horizontal_fov_deg:=360.0 -p vertical_fov_deg:=26.8 -p vertical_fov_up_deg:=2.0 -p vertical_fov_down_deg:=24.8 -p azimuth_0_to_hfov:=true"

# Terminal A: perturber
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P \
  -p rng_seed:=42

# Terminal B: estimator
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p publish_rel_low:=true

# Terminal C: roi_frame_logger
ros2 run kitti_roi_eval roi_frame_logger --ros-args \
  -p out_dir:=$OUT \
  -p save_gt_mask:=true -p save_rel_low:=true -p save_roi_masks:=true \
  -p save_format:=png -p save_npy:=false \
  -p split_masks_by_type:=true \
  -p save_vis_aligned:=true \
  -p vis_hfov_deg:=360.0 -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312 \
  -p vis_center_az:=true

# Terminal D: kitti_player_with_gt
ros2 run kitti_roi_eval kitti_player_with_gt --ros-args \
  -p drive_dir:=$DRIVE_DIR \
  -p points_topic:=/livox/lidar



python3 /home/agx-orin-07/ws_livox/eval/chap7/render_missing_overlay.py \
  --maps_dir $OUT \
  --use_rel_low \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_miss_vis_kitti.png \
  --frame_idx 200
```




### 図7.11〜7.12：統合出力（看板図 + 統合比較）
```bash
# dynamic bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/integration_eval/dynamic
export P=0.10
export V=128
export H=128
bash /home/agx-orin-07/ws_livox/eval/chap7/run_integration_eval.sh

ros2 run kitti_roi_eval roi_frame_logger --ros-args \
  -p out_dir:=/home/agx-orin-07/ws_livox/eval/chap7/integration_frames/dynamic \
  -p save_roi_masks:=true -p save_rel_low:=true \
  -p save_format:=png -p save_npy:=false \
  -p split_masks_by_type:=true \
  -p save_vis_aligned:=true \
  -p vis_hfov_deg:=360.0 -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312 \
  -p vis_center_az:=true

python3 /home/agx-orin-07/ws_livox/eval/chap7/render_integration_overlay.py \
  --maps_dir /home/agx-orin-07/ws_livox/eval/chap7/integration_frames/dynamic \
  --frame_idx 200 \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_integration_vis_dynamic.png

python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_integration_metrics.py \
  --imp_csv /home/agx-orin-07/ws_livox/eval/chap7/integration_eval/dynamic/p_0.10/imp/iou_per_frame.csv \
  --rel_csv /home/agx-orin-07/ws_livox/eval/chap7/integration_eval/dynamic/p_0.10/rel/iou_per_frame.csv \
  --int_csv /home/agx-orin-07/ws_livox/eval/chap7/integration_eval/dynamic/p_0.10/integrated/iou_per_frame.csv \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_cov_integrated_dynamic.png


# static bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/integration_eval/static
export P=0.10
export V=128
export H=128
bash /home/agx-orin-07/ws_livox/eval/chap7/run_integration_eval.sh
python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_integration_metrics.py \
  --imp_csv /home/agx-orin-07/ws_livox/eval/chap7/integration_eval/static/p_0.10/imp/iou_per_frame.csv \
  --rel_csv /home/agx-orin-07/ws_livox/eval/chap7/integration_eval/static/p_0.10/rel/iou_per_frame.csv \
  --int_csv /home/agx-orin-07/ws_livox/eval/chap7/integration_eval/static/p_0.10/integrated/iou_per_frame.csv \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_cov_integrated_static.png



# KITTI（bagなし）
source /opt/ros/humble/setup.bash
source ~/ws_livox/install/setup.bash

export DRIVE_DIR=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/integration_eval/kitti
export P=0.25
export V=128
export H=128
export KITTI_FOV_ARGS="-p horizontal_fov_deg:=360.0 -p vertical_fov_deg:=26.8 -p vertical_fov_up_deg:=2.0 -p vertical_fov_down_deg:=24.8 -p azimuth_0_to_hfov:=true"

# Terminal A: perturber
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:=$P

# Terminal B: estimator
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  ${KITTI_FOV_ARGS} \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p publish_rel_low:=true

# Terminal C/D/E: roi_eval_iou（3条件）
export GT_NPZ=/home/agx-orin-07/ws_livox/data/gt_masks/V128_H128/gt_binmask_V128_H128.npz

ros2 run kitti_roi_eval roi_cover_logger --ros-args \
  -p pred_topic:=roi_est/roi_imp_mono8 \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed \
  -p frame_idx_topic:=kitti_player/frame_idx \
  -p gt_npz:=$GT_NPZ \
  -p out_dir:=$OUT/imp \
  -p csv_name:=cover_per_frame.csv \
  -p align_use_gt_meta:=true \
  -p sync_policy:=strict_stamp \
  -p write_missing_tail_rows:=true \
  -p viz_enable:=true \
  -p viz_max_frames:=100000

ros2 run kitti_roi_eval roi_eval_iou --ros-args \
  -p sync_policy:=strict_stamp \
  -p frame_idx_mode:=topic \
  -p frame_idx_topic:=kitti_player/frame_idx \
  -p write_missing_tail_rows:=true \
  -p pred_topic:=roi_est/rel_low_mono8 \
  -p gt_topic:=pc_perturber/gt_mask_mono8 \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed \
  -p out_dir:=$OUT/rel -p csv_name:=iou_per_frame.csv \
  -p viz_enable:=true \
  -p viz_max_frames:=100000

ros2 run kitti_roi_eval roi_cover_logger --ros-args \
  -p sync_policy:=strict_stamp \
  -p pred_topic:=roi_est/rel_low_mono8 \
  -p pred_topic_aux:=roi_est/roi_imp_mono8 \
  -p pred_combine_mode:=and \
  -p gt_mask_topic:=pc_perturber/gt_mask_mono8 \
  -p gt_combine_mode:=and \
  -p omega_topic:=roi_est/omega_mono8 \
  -p pc_topic:=/livox/lidar_perturbed \
  -p frame_idx_topic:=kitti_player/frame_idx \
  -p gt_npz:=$GT_NPZ \
  -p out_dir:=$OUT/integrated \
  -p csv_name:=iou_per_frame.csv \
  -p align_use_gt_meta:=true \
  -p write_missing_tail_rows:=true \
  -p viz_enable:=true \
  -p viz_max_frames:=100000

# Terminal F: roi_frame_logger（KITTI向け横長vis保存）
export OUT_FRAMES=/home/agx-orin-07/ws_livox/eval/chap7/integration_frames/kitti

ros2 run kitti_roi_eval roi_frame_logger --ros-args \
  -p out_dir:=$OUT_FRAMES \
  -p save_enable:=true \
  -p save_roi_masks:=true \
  -p save_rel_low:=true \
  -p save_gt_bbox:=true \
  -p gt_npz_path:=$GT_NPZ \
  -p split_masks_by_type:=true \
  -p save_format:=png \
  -p save_npy:=false \
  -p save_vis_aligned:=true \
  -p vis_flip_ud:=true \
  -p vis_flip_lr:=true \
  -p vis_center_az:=true \
  -p vis_stretch:=true \
  -p vis_hfov_deg:=360.0 \
  -p vis_vfov_deg:=26.8 \
  -p vis_aspect_target:=3.312

# Terminal G: kitti_player_with_gt
ros2 run kitti_roi_eval kitti_player_with_gt --ros-args \
  -p drive_dir:=$DRIVE_DIR \
  -p points_topic:=/livox/lidar \
  -p rate_hz:=1.0


python3 /home/agx-orin-07/ws_livox/eval/chap7/render_integration_overlay.py \
  --maps_dir /home/agx-orin-07/ws_livox/eval/chap7/integration_frames/kitti \
  --frame_idx 200 \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_integration_vis_kitti.png

python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_integration_metrics.py \
  --imp_csv $OUT/imp/cover_per_frame.csv \
  --rel_csv $OUT/rel/iou_per_frame.csv \
  --int_csv $OUT/integrated/iou_per_frame.csv \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_cov_integrated_kitti.png
```


## GT可視化
ros2 run kitti_roi_eval kitti_player_with_gt --ros-args \
  -p drive_dir:=$DRIVE_DIR \
  -p points_topic:=/livox/lidar \
  -p rate_hz:=1.0 \
  -p publish_bbox_markers:=true \
  -p bbox_topic:=kitti_player/bbox_lines



### 図7.13〜7.14：処理時間分布 + 追従率
```bash
# dynamic bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/proc_time/dynamic
export V=128
export H=128
mkdir -p $OUT

ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=false \
  > $OUT/perturber.log 2>&1 &

ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p csv_enable:=true \
  -p csv_path:=$OUT/roi_est_proc_time.csv \
  -p csv_flush_every:=1 \
  > $OUT/estimator.log 2>&1 &

sleep 1
ros2 bag play $BAG --rate 1.0 --disable-keyboard-controls > $OUT/bagplay.log 2>&1

python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_proc_time.py \
  --csv /home/agx-orin-07/ws_livox/eval/chap7/proc_time/dynamic/roi_est_proc_time.csv \
  --period_ms 100 \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_proc_time_dynamic.png


# static bag
export BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_static_01
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/proc_time/static
export V=128
export H=128
mkdir -p $OUT
# Terminal A: perturber（dropなし）
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=false \
  > $OUT/perturber.log 2>&1 &
# Terminal B: estimator（proc time CSV）
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p csv_enable:=true \
  -p csv_path:=$OUT/roi_est_proc_time.csv \
  -p csv_flush_every:=1 \
  > $OUT/estimator.log 2>&1 &
# Terminal C: bag play
ros2 bag play $BAG --rate 1.0 --disable-keyboard-controls > $OUT/bagplay.log 2>&1
python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_proc_time.py \
  --csv /home/agx-orin-07/ws_livox/eval/chap7/proc_time/static/roi_est_proc_time.csv \
  --period_ms 100 \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_proc_time_static.png


# KITTI（bagなし）
export DRIVE_DIR=/home/agx-orin-07/ws_livox/data/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
export OUT=/home/agx-orin-07/ws_livox/eval/chap7/proc_time/kitti
export V=128
export H=128
mkdir -p $OUT

# Terminal A: perturber（dropなし）
ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
  -p input_topic:=/livox/lidar \
  -p output_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p enable_drop:=false \
  > $OUT/perturber.log 2>&1 &

# Terminal B: estimator（proc time CSV）
ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
  -p input_topic:=/livox/lidar_perturbed \
  -p num_vertical_bins:=$V -p num_horizontal_bins:=$H \
  -p csv_enable:=true \
  -p csv_path:=$OUT/roi_est_proc_time.csv \
  -p csv_flush_every:=1 \
  > $OUT/estimator.log 2>&1 &

# Terminal C: kitti_player_with_gt
ros2 run kitti_roi_eval kitti_player_with_gt --ros-args \
  -p drive_dir:=$DRIVE_DIR \
  -p points_topic:=/livox/lidar

  
python3 /home/agx-orin-07/ws_livox/eval/chap7/plot_proc_time.py \
  --csv /home/agx-orin-07/ws_livox/eval/chap7/proc_time/kitti/roi_est_proc_time.csv \
  --period_ms 100 \
  --out /home/agx-orin-07/ws_livox/eval/chap7/fig_proc_time_kitti.png
```

---

## 8. 主要トピックまとめ

### 重要 ROI 推定
- 入力: `/livox/lidar_perturbed` (PointCloud2)
- 出力:
  - `roi_est/importance_map` (Float32MultiArray)
  - `roi_est/rel_map` (Float32MultiArray)
  - `roi_est/roi_imp_mono8` / `roi_est/roi_use_mono8` / `roi_est/roi_alert_mono8`
  - `roi_est/rel_low_mono8` (mono8, `publish_rel_low=true`)
  - `roi_est/omega_mono8`
  - `roi_est/frame_rel`, `roi_est/frame_rel_obs`
  - `roi_est/alert_ratio`, `roi_est/alert_ratio_omega`

### 点群攪乱
- 入力: `/livox/lidar`
- 出力:
  - `/livox/lidar_perturbed`
  - `pc_perturber/gt_mask_mono8`
  - `pc_perturber/drop_ratio`
  - `pc_perturber/bias_m`

---

## 9. 既知の注意点 / トラブルシュート

- `roi_est_case*.yaml` に **古いパラメータ名** が含まれるため、  
  現行コードと一致していない場合は `important_roi_estimator.py` のパラメータ定義に合わせて修正してください。
- KITTI 用のパラメータ (`kitti_player.yaml`, `roi_eval_iou.yaml`) は **パスが固定文字列** になっているため、実環境に合わせて変更が必要です。
- `roi_min_logger` は **sim stamp** を前提に frame_idx を算出します。  
  `kitti_player_with_gt` の `use_sim_stamp=true` と `rate_hz` を一致させてください。

---

## 10. 次に読むべきコード

**入口:**
- `src/lidar_roi_nodes/lidar_roi_nodes/important_roi_estimator.py`
- `src/lidar_roi_nodes/lidar_roi_nodes/pointcloud_perturber.py`
- `src/lidar_edge_estimator/lidar_edge_estimator/angle_bin_edge_estimator.py`
- `src/kitti_roi_eval/kitti_roi_eval/roi_eval_iou.py`

---

## 11. ライセンス

`src/lidar_edge_estimator/package.xml` では MIT と記載されています。  
他パッケージは `TODO` になっているため、用途に応じて明記してください。
