"""Plot helpers (matplotlib is imported lazily)."""

from __future__ import annotations

import numpy as np


def plot_policy(policy, ax=None, y_max: float | None = None):
    """Plot the preemption map theta*(y) and the sampling target z*(Delta)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    y = policy.y
    m = y <= (y_max if y_max is not None else y[-1])
    ax[0].plot(y[m], policy.theta_map[m], lw=1.8)
    ax[0].set_xlabel("busy-start AoI $y$")
    ax[0].set_ylabel(r"preemption time $\theta^\star(y)$")
    ax[0].grid(alpha=0.3)
    ax[1].plot(y[m], policy.z_map[m], lw=1.8, label=r"$z^\star(\Delta)$")
    ax[1].plot(y[m], y[m], "k:", lw=1, label=r"$z=\Delta$ (no wait)")
    ax[1].set_xlabel(r"AoI at delivery $\Delta$")
    ax[1].set_ylabel("sampling target")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.3)
    ax[0].figure.tight_layout()
    return ax


def plot_sample_path(ts, ages, preempt_times=(), ax=None, label=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ts, ages, lw=1.4, label=label)
    for p in np.asarray(preempt_times):
        ax.axvline(p, color="tab:red", alpha=0.25, lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("AoI")
    ax.grid(alpha=0.3)
    return ax
