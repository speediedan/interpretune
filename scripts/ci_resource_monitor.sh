#!/usr/bin/env bash

INTERVAL="${CI_RESOURCE_MONITOR_INTERVAL:-5}"
ENABLED="${CI_RESOURCE_MONITOR:-1}"

if [[ "$ENABLED" != "1" ]]; then
  echo "[ci_resource_monitor] Monitoring disabled (CI_RESOURCE_MONITOR=$ENABLED)"
  exit 0
fi

echo "[ci_resource_monitor] Starting resource monitor (interval: ${INTERVAL}s)"
while true; do
  echo "==== [$(date)] Resource Usage ===="
  echo "-- Disk Usage --"
  df -h /
  echo "-- Memory Usage --"
  free -h || vm_stat || true
  echo "-- Top Processes (by CPU) --"
  ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 15
  echo "-- Top Processes (by MEM) --"
  ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 15
  echo "==============================="
  sleep "$INTERVAL"
done
