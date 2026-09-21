#!/bin/bash
PY=/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/.venv/bin/python
cd /Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/experiments/constrained_belief_updating/attn_only_redo/code
for cfg in A B; do
  $PY train_ao.py $cfg 1 1 --steps 12000
  $PY train_ao.py $cfg 1 2 --steps 12000
  $PY train_ao.py $cfg 2 2 --steps 12000
  $PY train_ao.py $cfg 3 2 --steps 12000
done
echo "ALL GATE CELLS DONE"
