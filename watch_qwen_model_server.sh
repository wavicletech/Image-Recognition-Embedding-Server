#!/bin/bash
# Cron health check. Three failures in a row while the process is alive means it is hung.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin"
STATE=/home/customer/qwen_embedding/logs/health_failures
mkdir -p /home/customer/qwen_embedding/logs
PORT="${QWEN_EMBED_PORT:-8090}"
if [ -f /home/customer/qwen_embedding/.env ]; then
  # shellcheck disable=SC1091
  port_line=$(grep -E '^QWEN_EMBED_PORT=' /home/customer/qwen_embedding/.env | tail -n 1 || true)
  if [ -n "$port_line" ]; then
    PORT=${port_line#QWEN_EMBED_PORT=}
  fi
fi

if ! /usr/bin/tmux has-session -t qwenmodelserver 2>/dev/null; then
  echo 0 > "$STATE"
  exit 0
fi

if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo 0 > "$STATE"
  exit 0
fi

pid=$(ps -eo pid,cmd | awk '/python -m app.server/ && !/awk/ {print $1; exit}')
if [ -z "$pid" ]; then
  echo 0 > "$STATE"
  exit 0
fi

count=0
if [ -f "$STATE" ]; then
  count=$(cat "$STATE" 2>/dev/null || echo 0)
fi
count=$((count + 1))
echo "$count" > "$STATE"
if [ "$count" -ge 3 ]; then
  echo "$(date -Is) embedding server hung; killing ${pid}" >> /home/customer/qwen_embedding/logs/server.log
  kill -9 "$pid" || true
  echo 0 > "$STATE"
fi
