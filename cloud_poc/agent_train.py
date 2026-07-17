"""Self-contained training driver for the cloud PoC.

Imports the sibling module `agent_model_task` (written in parallel) which
must expose:
    D = 4
    build_model(device) -> torch.nn.Module
    make_batch(batch_size, device) -> batch
    loss_fn(model, batch) -> scalar torch.Tensor

Runnable from anywhere: `python cloud_poc/agent_train.py`.
"""

import os
import sys
import warnings

# Suppress harmless numpy/torch UserWarnings so the loss curve stays readable.
warnings.filterwarnings("ignore", category=UserWarning)

# Fail loudly at import time if a core dependency is missing.
import torch  # noqa: E402
import transformer_lens  # noqa: E402  (imported to surface missing-dep errors early)

# Ensure this file's directory is on sys.path so `import agent_model_task`
# works regardless of the current working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import agent_model_task  # noqa: E402


BATCH_SIZE = 256
NUM_STEPS = 500
LEARNING_RATE = 1e-3
RECORD_STEPS = (1, 100, 250, 500)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device {device}")

    torch.manual_seed(0)

    model = agent_model_task.build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    recorded = {}

    for step in range(1, NUM_STEPS + 1):
        batch = agent_model_task.make_batch(BATCH_SIZE, device)
        loss = agent_model_task.loss_fn(model, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in RECORD_STEPS:
            val = float(loss.detach())
            recorded[step] = val
            print(f"step {step} loss {val:.6f}")

    summary = "LOSSCURVE " + " ".join(
        f"step{n}={recorded[n]:.6f}" for n in RECORD_STEPS
    )
    print(summary)


if __name__ == "__main__":
    main()
