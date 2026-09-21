#!/bin/bash
PY=/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/.venv/bin/python
cd /Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/experiments/constrained_belief_updating/hierarchy_fra/code
until grep -q "GATE DONE" ../out/gate_run.log 2>/dev/null; do sleep 10; done
echo "noisy pair done; running clean-obs pair"
$PY hier_gate.py --tag H_gated_clean --gated 1 --sigma 0.05 --steps 8000
$PY hier_gate.py --tag H_null_clean  --gated 0 --sigma 0.05 --steps 8000
echo "CLEAN GATE DONE"
