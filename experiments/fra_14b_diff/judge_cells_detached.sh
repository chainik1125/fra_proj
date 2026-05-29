#!/bin/bash
PY=/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/.venv/bin/python3
cd /tmp
LOG=/tmp/judge_routing_all.log
echo "=== routing judge batch start $(date -u +%H:%M:%S) ===" > $LOG
for cell in \
  qk_to_ov_finegrid/frarouting_qk_to_ov_ln1_gran1 \
  qk_to_qk_finegrid/frarouting_qk_to_qk_ln1_gran1 \
  ov_to_ov_finegrid/frarouting_ov_to_ov_ln1_gran1 ; do
    echo ">>> $(date -u +%H:%M:%S) judging $cell" >> $LOG
    "$PY" /tmp/judge_one_cell.py "$cell" >> $LOG 2>&1
    echo "<<< $(date -u +%H:%M:%S) exit=$? for $cell" >> $LOG
    safe=$(echo "$cell" | tr '/' '_')
    cp /tmp/judge_one_cell.log "/tmp/judge_${safe}.log" 2>/dev/null
done
echo "=== routing judge batch DONE $(date -u +%H:%M:%S) ===" >> $LOG
