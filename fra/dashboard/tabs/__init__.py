"""Dashboard tab modules -- each exposes a ``render(tab)`` function."""

from fra.dashboard.tabs import ablation, di, interactions, matrix, max_act, reconstruction

__all__ = ["ablation", "di", "interactions", "matrix", "max_act", "reconstruction"]
