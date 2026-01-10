#!/usr/bin/env bash
set -euo pipefail

# --- ROS setup（この間だけnounset無効）---
set +u
source /opt/ros/humble/setup.bash
source ~/ws_livox/install/setup.bash
set -u

# ==========
# 設定
# ==========
QS=(0.00 0.05 0.10 0.15 0.20 0.25)

BAG=/home/agx-orin-07/ws_livox/bags/20251222_orin_avia_in_dynamic_01
OUT=/home/agx-orin-07/ws_livox/eval/drop_sweep

RATE=1.0
PLAY_TIMEOUT_SEC=90

STARTUP_SLEEP_SEC=1.0

# perturber
PACKET_POINTS=100
RNG_SEED=1234

# estimator
ROI_TOP_PERCENT=10.0
TAU_REL=0.6

# CSV flush（重要）
CSV_FLUSH_EVERY=1

# 画像スナップ（必要なければ SNAP_Q="" に）
SNAP_Q="0.10"
SNAP_FRAME=200
SAVE_START=$((SNAP_FRAME-5))
SAVE_STOP=$((SNAP_FRAME+5))
if [ "$SAVE_START" -lt 1 ]; then SAVE_START=1; fi

mkdir -p "$OUT"

# ==========
# 残骸掃除（スイープ前に毎回やる）
# ==========
cleanup_procs() {
  pkill -INT -f "lidar_roi_nodes.*pointcloud_perturber" 2>/dev/null || true
  pkill -INT -f "lidar_roi_nodes.*important_roi_estimator" 2>/dev/null || true
  sleep 0.3
  pkill -KILL -f "lidar_roi_nodes.*pointcloud_perturber" 2>/dev/null || true
  pkill -KILL -f "lidar_roi_nodes.*important_roi_estimator" 2>/dev/null || true
}

echo "[INFO] OUT=$OUT"
echo "[INFO] BAG=$BAG"
echo "[INFO] QS=(${QS[*]})"
echo "[INFO] SNAP_Q=$SNAP_Q SNAP_FRAME=$SNAP_FRAME (range ${SAVE_START}..${SAVE_STOP})"

# bag存在チェック
if [ ! -f "$BAG/metadata.yaml" ]; then
  echo "[ERROR] BAG folder not found or metadata.yaml missing: $BAG"
  exit 1
fi

for Q in "${QS[@]}"; do
  SNAP_DIR="$OUT/snapshot_q_${Q}"
  mkdir -p "$SNAP_DIR/images"

  echo "======================================"
  echo "=== q=${Q} ==="
  echo "SNAP_DIR=$SNAP_DIR"
  echo "======================================"

  # 前回の残骸を消す（ここが効きます）
  cleanup_procs

  # このqで画像保存するか
  SAVE_ENABLE=false
  SAVE_RANGE_ARGS=()
  if [[ -n "${SNAP_Q}" && "${Q}" == "${SNAP_Q}" ]]; then
    SAVE_ENABLE=true
    SAVE_RANGE_ARGS=(
      -p save_start_frame:="$SAVE_START"
      -p save_stop_frame:="$SAVE_STOP"
      -p save_every_n:=1
      -p save_format:=png
      -p save_importance:=true
      -p save_roi_rgb:=true
      -p save_range:=true
      -p save_reliability:=false
      -p save_roi_masks:=false
      -p save_npy:=false
    )
    echo "[INFO] snapshot enabled for q=${Q}"
  fi

  # CSVを毎回作り直す（古いヘッダだけ残る事故を避ける）
  rm -f "$SNAP_DIR/frames.csv"

  # --- perturber（timeoutで包む：残プロセス防止）---
  ( timeout --signal=INT --kill-after=3s "${PLAY_TIMEOUT_SEC}s" \
      ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
        -p input_topic:=/livox/lidar \
        -p output_topic:=/livox/lidar_perturbed \
        -p sub_reliability:=best_effort -p pub_reliability:=best_effort \
        -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:="$Q" \
        -p packet_points:="$PACKET_POINTS" -p rng_seed:="$RNG_SEED" \
        -p enable_spoof:=false \
      > "$SNAP_DIR/perturber.log" 2>&1
  ) &
  PERT_PID=$!

  # --- estimator（timeoutで包む + csv_flush_every）---
  ( timeout --signal=INT --kill-after=3s "${PLAY_TIMEOUT_SEC}s" \
      ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
        -p input_topic:=/livox/lidar_perturbed \
        -p sub_reliability:=best_effort -p pub_reliability:=best_effort \
        -p metrics_enable:=false \
        -p roi_top_percent:="$ROI_TOP_PERCENT" \
        -p tau_rel:="$TAU_REL" \
        -p csv_enable:=true \
        -p csv_path:="$SNAP_DIR/frames.csv" \
        -p csv_flush_every:="$CSV_FLUSH_EVERY" \
        -p save_enable:="$SAVE_ENABLE" \
        -p save_dir:="$SNAP_DIR/images" \
        "${SAVE_RANGE_ARGS[@]}" \
      > "$SNAP_DIR/estimator.log" 2>&1
  ) &
  EST_PID=$!

  sleep "$STARTUP_SLEEP_SEC"

  # --- bag play（必ず disable-keyboard-controls を付ける）---
  echo "[INFO] playing bag..."
  timeout --signal=INT --kill-after=5s "${PLAY_TIMEOUT_SEC}s" \
    ros2 bag play "$BAG" --rate "$RATE" --disable-keyboard-controls \
    > "$SNAP_DIR/bagplay.log" 2>&1 || true
  echo "[INFO] bag play done (or timeout)."

  # 少し待ってCSV/画像の書き込みを落ち着かせる
  sleep 0.5

  # --- ノード停止（timeoutラッパPIDにINT）---
  kill -INT "$EST_PID" 2>/dev/null || true
  kill -INT "$PERT_PID" 2>/dev/null || true
  wait "$EST_PID" 2>/dev/null || true
  wait "$PERT_PID" 2>/dev/null || true

  # --- 成功判定（frames.csvが増えているか）---
  LINES=$(wc -l < "$SNAP_DIR/frames.csv" 2>/dev/null || echo 0)
  if [ "$LINES" -le 1 ]; then
    echo "[WARN] frames.csv has no data rows (lines=$LINES). Showing last logs:"
    echo "----- estimator.log (tail) -----"
    tail -n 50 "$SNAP_DIR/estimator.log" || true
    echo "----- perturber.log (tail) -----"
    tail -n 50 "$SNAP_DIR/perturber.log" || true
    echo "----- bagplay.log (tail) -----"
    tail -n 50 "$SNAP_DIR/bagplay.log" || true
  else
    echo "[OK] frames.csv updated (lines=$LINES)"
  fi

  echo "[INFO] finished q=${Q}"
done

echo "[INFO] sweep complete. OUT=$OUT"

