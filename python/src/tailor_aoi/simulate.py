"""Monte Carlo evaluation of sampling/preemption policies (SMDP-consistent).

Port of ``simulate_aoi_smdp_strict_q99`` in ``Paretov2.m``: after a delivery
the source waits until the AoI reaches the sampling target and pays
``kappa_s``; a service still running after ``theta(y)`` is preempted, costs
``kappa_p`` and restarts immediately with the same AoI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import ServiceDist


@dataclass
class SimResult:
    avg_cost: float
    avg_aoi: float
    sampling_rate: float
    preemption_rate: float
    q99_aoi: float
    horizon: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"avg cost {self.avg_cost:.5f} | avg AoI {self.avg_aoi:.5f} | "
                f"q99 AoI {self.q99_aoi:.4f} | samples/t {self.sampling_rate:.4f} | "
                f"preemptions/t {self.preemption_rate:.4f}")


def _lookup(policy):
    """Return fast scalar callables z(delta), theta(y) for the given policy."""
    if getattr(policy, "kind", "") == "TAILOR":
        zm, tm, dt = policy.z_map, policy.theta_map, policy.dt
        nz, far = zm.size, policy.rho - policy.slope
        z_list, t_list = zm.tolist(), tm.tolist()

        def z(d):
            i = int(round(d / dt))
            return z_list[i] if i < nz else max(d, far)

        def th(y):
            i = int(round(y / dt))
            return t_list[i] if i < nz else t_list[-1]

        return z, th
    D, T = float(policy.D), float(policy.T)
    return (lambda d: d if d > D else D), (lambda y: T)


def simulate(policy, dist: ServiceDist, kappa_s: float = 1.0, kappa_p: float = 1.0,
             n_attempts: int = 500_000, rng=None, samples: np.ndarray | None = None) -> SimResult:
    """Simulate ``n_attempts`` service attempts and return time averages.

    Pass the same ``samples`` array to compare several policies under common
    random numbers (as done in the paper's MATLAB code).
    """
    S_vec = samples if samples is not None else dist.sample(n_attempts, rng)
    S_list = np.asarray(S_vec, float).tolist()
    z_of, th_of = _lookup(policy)

    t_now = area = 0.0
    n_samp = n_pre = 0
    in_idle, delta, y = True, 0.0, 0.0
    seg_a, seg_b = [], []
    for S in S_list:
        if in_idle:
            z = z_of(delta)
            w = z - delta if z > delta else 0.0
            if w > 0:
                seg_a.append(delta); seg_b.append(delta + w)
            area += delta * w + 0.5 * w * w
            t_now += w
            n_samp += 1
            y = delta + w
            in_idle = False
        th = th_of(y)
        dur = S if S <= th else th
        if dur > 0:
            seg_a.append(y); seg_b.append(y + dur)
        area += y * dur + 0.5 * dur * dur
        t_now += dur
        if S <= th:
            delta, in_idle = S, True
        else:
            n_pre += 1
            y += dur
    avg_aoi = area / t_now
    return SimResult(
        avg_cost=(area + kappa_s * n_samp + kappa_p * n_pre) / t_now,
        avg_aoi=avg_aoi,
        sampling_rate=n_samp / t_now,
        preemption_rate=n_pre / t_now,
        q99_aoi=_time_weighted_quantile(np.array(seg_a), np.array(seg_b), 0.99),
        horizon=t_now,
    )


def _time_weighted_quantile(a, b, p):
    """Quantile of the AoI under the time-average distribution (exact sweep)."""
    L = b - a
    ok = (L > 0) & np.isfinite(a) & np.isfinite(b)
    a, b, L = a[ok], b[ok], L[ok]
    total = L.sum()
    if total <= 0:
        return float("nan")
    x = np.concatenate([a, b])
    typ = np.concatenate([np.ones(a.size), -np.ones(b.size)])
    order = np.lexsort((typ, x))
    x, typ = x[order], typ[order]
    active = np.cumsum(typ)  # number of active segments right after each event
    dx = np.diff(x)
    mass = np.concatenate([[0.0], np.cumsum(active[:-1] * dx) / total])
    k = np.searchsorted(mass, p)
    if k >= mass.size:
        return float(b.max())
    k0 = k - 1
    c = active[k0]
    return float(x[k0] + (p - mass[k0]) * total / c) if c > 0 else float(x[k])


def sample_path(policy, dist: ServiceDist, horizon: float = 50.0, rng=None):
    """Return (time, AoI) breakpoints of one sample path, for plotting."""
    rng = np.random.default_rng(rng)
    z_of, th_of = _lookup(policy)
    t, age = 0.0, 0.0
    ts, ages, pre = [0.0], [0.0], []
    while t < horizon:
        w = max(0.0, z_of(age) - age)
        if w > 0:
            t += w; age += w
            ts.append(t); ages.append(age)
        while True:
            S = float(dist.sample(1, rng)[0])
            th = th_of(age)
            if S <= th:
                t += S; ts.append(t); ages.append(age + S)
                age = S; ts.append(t); ages.append(age)
                break
            t += th; age += th
            ts.append(t); ages.append(age); pre.append(t)
    return np.array(ts), np.array(ages), np.array(pre)
