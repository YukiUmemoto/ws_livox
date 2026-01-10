#!/usr/bin/env bash
set -euo pipefail

# =========================
# 設定
# =========================
Q_SNAP=0.10
BAG=~/ws_livox/bags/20251222_orin_avia_in_dynamic_01

OUT=~/ws_livox/eval/drop_sweep
SNAP_DIR="$OUT/snapshot_q_$(printf "%.2f" "$Q_SNAP")"
IMG_DIR="$SNAP_DIR/images"
CSV_PATH="$SNAP_DIR/frames_snapshot.csv"

SUB_REL=best_effort
PUB_REL=best_effort

ROI_TOP_PERCENT=10.0
TAU_REL=0.6

# 保存したい中心フレーム（1始まり）
SAVE_FRAME=200
SAVE_START=$((SAVE_FRAME-5))
SAVE_STOP=$((SAVE_FRAME+5))
if [ "$SAVE_START" -lt 1 ]; then SAVE_START=1; fi

RATE=1.0

# 「フレーム到達」を見て止めるが、念のための上限（秒）
MAX_WAIT_SEC=180

# ノード停止待ち上限（秒）
STOP_TIMEOUT_SEC=5

# 既存の同名プロセスがいたら強制停止する（実験用：安全側）
FORCE_KILL_OLD=true

# =========================
# 便利関数
# =========================
log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err() { echo "[ERROR] $*" >&2; }

# setsid で「新しいセッション/PGID」を作って起動し、PID(=PGID)を返す
start_bg() {
  local name="$1"; shift
  local logfile="$1"; shift
  setsid "$@" > "$logfile" 2>&1 &
  local pid=$!
  log "started $name pid=$pid log=$logfile"
  echo "$pid"
}

# プロセスグループごと止める（ros2 run が子プロセスを持つ問題対策）
stop_pgid() {
  local pid="${1:-}"
  local name="${2:-proc}"
  local timeout_sec="${3:-5}"

  if [ -z "$pid" ]; then return 0; fi
  if ! kill -0 "$pid" 2>/dev/null; then return 0; fi

  log "stopping $name (pgid=$pid) ..."
  kill -INT -- "-$pid" 2>/dev/null || true

  local t=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 0.1
    t=$((t+1))
    if [ "$t" -ge $((timeout_sec*10)) ]; then
      warn "$name still alive -> KILL (pgid=$pid)"
      kill -KILL -- "-$pid" 2>/dev/null || true
      break
    fi
  done
}

# CSV の最終 frame_idx を取る（ヘッダのみ/未作成なら 0）
get_last_frame_idx() {
  if [ ! -f "$CSV_PATH" ]; then
    echo 0; return
  fi
  local n
  n="$(wc -l < "$CSV_PATH" | tr -d ' ')"
  if [ "$n" -lt 2 ]; then
    echo 0; return
  fi
  local last
  last="$(tail -n 1 "$CSV_PATH" | cut -d, -f1 | tr -d ' ')"
  if [[ "$last" =~ ^[0-9]+$ ]]; then
    echo "$last"
  else
    echo 0
  fi
}

cleanup() {
  # 逆順で止める
  stop_pgid "${BAG_PID:-}"  "ros2 bag play"           "$STOP_TIMEOUT_SEC" || true
  stop_pgid "${EST_PID:-}"  "important_roi_estimator" "$STOP_TIMEOUT_SEC" || true
  stop_pgid "${PERT_PID:-}" "pointcloud_perturber"    "$STOP_TIMEOUT_SEC" || true
}
trap cleanup EXIT INT TERM

# =========================
# 前準備
# =========================
mkdir -p "$IMG_DIR"

if [ ! -f "$BAG/metadata.yaml" ]; then
  err "BAG folder not found or metadata.yaml missing: $BAG"
  exit 1
fi

log "SNAP_DIR=$SNAP_DIR"
log "Q_SNAP=$Q_SNAP"
log "BAG=$BAG"
log "SAVE_FRAME=$SAVE_FRAME (save range: $SAVE_START..$SAVE_STOP)"
log "RATE=$RATE MAX_WAIT_SEC=$MAX_WAIT_SEC"

# 既存プロセスの残骸があると「入力が流れない/混線」になりやすいので、必要なら殺す
log "existing related processes (if any):"
pgrep -af "ros2 bag play|pointcloud_perturber|important_roi_estimator" || true

if [ "$FORCE_KILL_OLD" = true ]; then
  log "FORCE_KILL_OLD=true -> try stopping old processes..."
  pkill -INT  -f "lidar_roi_nodes.*pointcloud_perturber" 2>/dev/null || true
  pkill -INT  -f "lidar_roi_nodes.*important_roi_estimator" 2>/dev/null || true
  pkill -INT  -f "ros2 bag play" 2>/dev/null || true
  sleep 0.5
  pkill -KILL -f "lidar_roi_nodes.*pointcloud_perturber" 2>/dev/null || true
  pkill -KILL -f "lidar_roi_nodes.*important_roi_estimator" 2>/dev/null || true
  pkill -KILL -f "ros2 bag play" 2>/dev/null || true
  sleep 0.2
fi

# 出力が「更新されない」混乱を避ける：既存CSVは退避、該当フレーム画像は削除
if [ -f "$CSV_PATH" ]; then
  mv "$CSV_PATH" "$CSV_PATH.bak_$(date +%Y%m%d_%H%M%S)"
  log "moved existing CSV to backup"
fi

for f in $(seq "$SAVE_START" "$SAVE_STOP"); do
  printf -v ff "%06d" "$f"
  rm -f "$IMG_DIR/frame_${ff}_"* 2>/dev/null || true
done

# =========================
# 起動（perturber -> estimator -> bag play）
# =========================
PERT_PID="$(start_bg "pointcloud_perturber" "$SNAP_DIR/perturber_snapshot.log" \
  ros2 run lidar_roi_nodes pointcloud_perturber --ros-args \
    -p input_topic:=/livox/lidar \
    -p output_topic:=/livox/lidar_perturbed \
    -p sub_reliability:="$SUB_REL" -p pub_reliability:="$PUB_REL" \
    -p enable_drop:=true -p drop_mode:=random -p drop_prob_q:="$Q_SNAP" \
    -p packet_points:=100 -p rng_seed:=1234 \
    -p enable_spoof:=false
)"

EST_PID="$(start_bg "important_roi_estimator" "$SNAP_DIR/estimator_snapshot.log" \
  ros2 run lidar_roi_nodes important_roi_estimator --ros-args \
    -p input_topic:=/livox/lidar_perturbed \
    -p sub_reliability:="$SUB_REL" -p pub_reliability:="$PUB_REL" \
    -p metrics_enable:=false \
    -p roi_top_percent:="$ROI_TOP_PERCENT" \
    -p tau_rel:="$TAU_REL" \
    -p csv_enable:=true \
    -p csv_path:="$CSV_PATH" \
    #-p csv_flush_every:=1 \
    -p save_enable:=true \
    -p save_dir:="$IMG_DIR" \
    -p save_start_frame:="$SAVE_START" \
    -p save_stop_frame:="$SAVE_STOP" \
    -p save_every_n:=1 \
    -p save_format:=png \
    -p save_importance:=true \
    -p save_roi_rgb:=true \
    -p save_range:=true \
    -p save_reliability:=false \
    -p save_roi_masks:=false \
    -p save_npy:=false
)"

sleep 1.0

BAG_PID="$(start_bg "ros2 bag play" "$SNAP_DIR/bagplay_snapshot.log" \
  ros2 bag play "$BAG" --rate "$RATE" --disable-keyboard-controls
)"

# =========================
# 進捗待ち：CSVの frame_idx を見て SAVE_STOP 到達で止める
# =========================
deadline=$((SECONDS + MAX_WAIT_SEC))
last=0

while true; do
  # bag play が落ちていたら打ち切り
  if ! kill -0 "$BAG_PID" 2>/dev/null; then
    warn "bag play exited before reaching SAVE_STOP. last_frame=$last"
    break
  fi

  last="$(get_last_frame_idx)"
  # たまに表示
  if (( SECONDS % 3 == 0 )); then
    log "progress: last_frame=$last / target=$SAVE_STOP"
  fi

  if [ "$last" -ge "$SAVE_STOP" ]; then
    log "reached target frame: $last >= $SAVE_STOP"
    break
  fi

  if [ "$SECONDS" -ge "$deadline" ]; then
    err "timeout: did not reach SAVE_STOP=$SAVE_STOP within MAX_WAIT_SEC=$MAX_WAIT_SEC (last=$last)"
    err "diagnostic: check topic flow during run:"
    err "  ros2 topic hz /livox/lidar"
    err "  ros2 topic hz /livox/lidar_perturbed"
    break
  fi

  sleep 0.2
done

# 余韻（ファイル書き込みのため）
sleep 1.0

# 明示停止（trap cleanup でも止まるが、ここで確実に止める）
stop_pgid "$BAG_PID"  "ros2 bag play"           "$STOP_TIMEOUT_SEC" || true
stop_pgid "$EST_PID"  "important_roi_estimator" "$STOP_TIMEOUT_SEC" || true
stop_pgid "$PERT_PID" "pointcloud_perturber"    "$STOP_TIMEOUT_SEC" || true

# =========================
# 結果サマリ
# =========================
log "csv lines: $(wc -l < "$CSV_PATH" 2>/dev/null || echo 0)  (expect >= 2)"
log "saved png count in range:"
cnt=0
for f in $(seq "$SAVE_START" "$SAVE_STOP"); do
  printf -v ff "%06d" "$f"
  c="$(ls -1 "$IMG_DIR"/frame_"$ff"_*.png 2>/dev/null | wc -l | tr -d ' ')"
  cnt=$((cnt + c))
done
log "  total png in [$SAVE_START..$SAVE_STOP] = $cnt"
log "done: $IMG_DIR"

