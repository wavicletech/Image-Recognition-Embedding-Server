#!/bin/bash
# Restart the embedding server if the process exits. The tmux session stays up.
set -u
export HOME=/home/customer
export PATH="/usr/local/bin:/usr/bin:/bin"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd /home/customer/qwen_embedding
set -a
# shellcheck disable=SC1091
source /home/customer/qwen_embedding/.env
set +a
export QWEN_EMBED_HOST="${QWEN_EMBED_HOST:-0.0.0.0}"
export QWEN_EMBED_PORT="${QWEN_EMBED_PORT:-8090}"
PY=/home/customer/qwen_embedding/.conda/bin/python
LOG_DIR=/home/customer/qwen_embedding/logs
mkdir -p "$LOG_DIR"

wait_for_gpu() {
  local attempt
  for attempt in $(seq 1 60); do
    if nvidia-smi >/dev/null 2>&1; then
      return 0
    fi
    echo "$(date -Is) waiting for the GPU driver"
    sleep 2
  done
  echo "$(date -Is) GPU driver did not respond"
  return 1
}

rotate_log() {
  local file="$LOG_DIR/server.log"
  local size=0
  if [ -f "$file" ]; then
    size=$(stat -c%s "$file")
  fi
  if [ "$size" -lt 20971520 ]; then
    return 0
  fi
  rm -f "$LOG_DIR/server.log.5"
  local index
  for index in 4 3 2 1; do
    if [ -f "$LOG_DIR/server.log.$index" ]; then
      mv "$LOG_DIR/server.log.$index" "$LOG_DIR/server.log.$((index + 1))"
    fi
  done
  mv "$file" "$LOG_DIR/server.log.1"
}

delay=5
while true; do
  wait_for_gpu || true
  rotate_log
  echo "$(date -Is) starting qwen embedding server"
  start_ts=$(date +%s)
  set +e
  set -o pipefail
  "$PY" -m app.server 2>&1 | tee -a "$LOG_DIR/server.log"
  code=${PIPESTATUS[0]}
  set +o pipefail
  set -e
  runtime=$(( $(date +%s) - start_ts ))
  if [ "$runtime" -ge 600 ]; then
    delay=5
  fi
  echo "$(date -Is) qwen embedding server exited (${code}) after ${runtime}s, restarting in ${delay}s"
  sleep "$delay"
  if [ "$runtime" -lt 60 ]; then
    delay=$((delay * 2))
    if [ "$delay" -gt 60 ]; then
      delay=60
    fi
  fi
done
