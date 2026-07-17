#!/bin/bash
# Run ON the freshly-provisioned L40S pod (pytorch image already has torch+numpy).
# Installs the extra deps for the synthetic + real-EM experiments.
set -e
pip install -q numpy transformers peft pyyaml matplotlib 2>&1 | tail -2
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "pod bootstrap done"
