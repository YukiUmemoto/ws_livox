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
  --V 128 --H 128 --hfov_deg 360 --vfov_deg 60
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

## 8. 主要トピックまとめ

### 重要 ROI 推定
- 入力: `/livox/lidar_perturbed` (PointCloud2)
- 出力:
  - `roi_est/importance_map` (Float32MultiArray)
  - `roi_est/rel_map` (Float32MultiArray)
  - `roi_est/roi_imp_mono8` / `roi_est/roi_use_mono8` / `roi_est/roi_alert_mono8`
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
