"""Minimal self-contained training driver for the cloud PoC.

Trains against the interface exposed by model_task.py:
    D = 4
    build_model(device) -> torch.nn.Module
    make_batch(batch_size, device) -> <batch>
    loss_fn(model, batch) -> torch.Tensor (scalar)
"""
import torch


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_task import build_model, make_batch, loss_fn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"torch {torch.__version__} | device {device}")

    torch.manual_seed(0)

    batch_size = 256
    n_steps = 500
    record_steps = (1, 100, 250, 500)
    recorded = {}

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, n_steps + 1):
        batch = make_batch(batch_size, device)
        loss = loss_fn(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in record_steps:
            recorded[step] = float(loss.detach())
            print(f"step {step:4d}  loss {recorded[step]:.6f}")

    print(
        "LOSSCURVE "
        f"step1={recorded.get(1, float('nan')):.6f} "
        f"step100={recorded.get(100, float('nan')):.6f} "
        f"step250={recorded.get(250, float('nan')):.6f} "
        f"step500={recorded.get(500, float('nan')):.6f}"
    )


if __name__ == "__main__":
    main()
