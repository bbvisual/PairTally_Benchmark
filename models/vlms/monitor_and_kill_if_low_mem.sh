#!/bin/bash
# Monitor free memory and kill evaluation script if free RAM < 10GB
THRESHOLD=10000  # 10GB in MB
INTERVAL=10      # seconds between checks

while true; do
  free_mem=$(free -m | awk '/Mem:/ {print $4}')
  if [ "$free_mem" -lt "$THRESHOLD" ]; then
    echo "[MONITOR] Low memory ($free_mem MB)! Killing evaluation script." | tee -a monitor_and_kill.log
    pkill -f evaluate_DICTA25_internvl3.py
    break
  fi
  sleep $INTERVAL
done 