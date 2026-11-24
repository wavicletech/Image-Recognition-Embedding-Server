#!/bin/bash
# Start the embedding server in a detached tmux session. No-op if it is already up.
set -u
export HOME=/home/customer
export PATH="/usr/bin:/bin"
/home/customer/qwen_embedding/watch_qwen_model_server.sh
SESSION=qwenmodelserver
if /usr/bin/tmux has-session -t "$SESSION" 2>/dev/null; then
  if [ -t 1 ]; then
    echo "qwenmodelserver is already running"
  fi
  exit 0
fi
/usr/bin/tmux new-session -d -s "$SESSION" /home/customer/qwen_embedding/run_qwen_loop.sh
echo "started qwenmodelserver"
