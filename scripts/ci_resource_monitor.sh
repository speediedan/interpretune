#!/usr/bin/env bash

INTERVAL="${CI_RESOURCE_MONITOR_INTERVAL:-5}"
while true; do
  echo "==== [$(date '+%Y-%m-%d %H:%M:%S.%3N')] Resource Usage ===="
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
