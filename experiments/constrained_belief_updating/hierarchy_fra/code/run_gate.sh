#!/bin/bash
PY=/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/.venv/bin/python
cd /Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/experiments/constrained_belief_updating/hierarchy_fra/code
$PY hier_gate.py --tag H_gated --gated 1 --sigma 0.6 --steps 8000
$PY hier_gate.py --tag H_null  --gated 0 --sigma 0.6 --steps 8000
echo "GATE DONE"
