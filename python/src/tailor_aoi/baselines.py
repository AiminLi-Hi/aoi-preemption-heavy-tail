"""Benchmark policies with exact (renewal-reward) average costs.

A *static* policy ``(D, T)`` waits until the AoI reaches ``D`` before sampling
and preempts any service that is still running after ``T`` time units.  Its
long-run average cost (time-average AoI plus sampling/preemption costs per
unit time) follows from the renewal-reward theorem over delivery cycles:

* the AoI at the start of a cycle is a completed service time ``S ~ Y | Y <= T``;
* the cycle waits ``W = (D - S)^+`` and then runs ``N`` failed attempts of
  length ``T`` (geometric, success probability ``F(T)``) and one successful
  attempt ``S' ~ Y | Y <= T``.

Special cases: Zero-Wait is ``(0, inf)``; Update-or-Wait is ``(beta, inf)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .distributions import ServiceDist


@dataclass(frozen=True)
class StaticPolicy:
    """Wait until AoI >= ``D``, then sample; preempt after ``T`` (``inf`` = never)."""

    D: float
    T: float
    name: str = "Static"
    kind: str = "Static"

    def sampling_target(self, delta):
        return np.maximum(np.asarray(delta, float), self.D)

    def wait_time(self, delta):
        return np.maximum(0.0, self.D - np.asarray(delta, float))

    def preemption_time(self, y):
        return np.full(np.shape(y), self.T, dtype=float)


def static_cost(dist: ServiceDist, D: float, T: float, kappa_s: float = 1.0, kappa_p: float = 1.0) -> float:
    """Exact long-run average cost of the static policy ``(D, T)``."""
    D = max(float(D), 0.0)
    T = float(T)
    p = dist.partial_moment(0, T)  # P(Y <= T)
    if p <= 0:
        return np.inf
    ES = dist.partial_moment(1, T) / p
    ES2 = dist.partial_moment(2, T) / p
    a = min(D, T)
    P_a, m1_a, m2_a = (dist.partial_moment(k, a) for k in range(3))
    EW = (D * P_a - m1_a) / p
    idle_area = (D**2 * P_a - m2_a) / (2 * p)  # E[S W + W^2/2]
    Ey0 = ES + EW
    if np.isinf(T):
        EN = EN2 = 0.0
        EB, EB2 = ES, ES2
    else:
        EN = (1 - p) / p
        EN2 = (1 - p) * (2 - p) / p**2
        EB = T * EN + ES
        EB2 = T**2 * EN2 + 2 * T * EN * ES + ES2
    num = idle_area + Ey0 * EB + 0.5 * EB2 + kappa_s + kappa_p * EN
    return float(num / (EW + EB))


def zero_wait(dist: ServiceDist, kappa_s: float = 1.0, kappa_p: float = 1.0):
    """Zero-Wait without preemption: sample right after every delivery."""
    pol = StaticPolicy(0.0, np.inf, name="Zero-Wait")
    return pol, static_cost(dist, 0.0, np.inf, kappa_s, kappa_p)


def update_or_wait(dist: ServiceDist, kappa_s: float = 1.0, kappa_p: float = 1.0, rule: str = "sun"):
    """Update-or-Wait without preemption (threshold policy ``z = max(Delta, beta)``).

    ``rule="sun"``   : the AoI-optimal threshold of Sun et al. (IEEE TIT 2017),
                       i.e. the root of E[X] = E[X^2] / (2 beta), X = max(Y, beta).
                       This is the benchmark used in the ISIT 2026 paper.
    ``rule="cost"``  : the threshold that minimises the average cost *including*
                       the sampling cost, i.e. eq. (29) of the paper.
    """
    if rule == "sun":
        def o(beta):
            Fb = dist.partial_moment(0, beta)
            EX = beta * Fb + dist.mean - dist.partial_moment(1, beta)
            EX2 = beta**2 * Fb + dist.second_moment - dist.partial_moment(2, beta)
            return EX - EX2 / (2 * beta)

        hi = max(dist.mean, 1.0)
        while o(hi) < 0:
            hi *= 2
        beta = optimize.brentq(o, 1e-12, hi, xtol=1e-12)
    elif rule == "cost":
        res = optimize.minimize_scalar(
            lambda b: static_cost(dist, b, np.inf, kappa_s, kappa_p),
            bounds=(0.0, 50 * dist.mean + dist.second_moment / max(dist.mean, 1e-12)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        beta = float(res.x)
    else:
        raise ValueError("rule must be 'sun' or 'cost'")
    pol = StaticPolicy(beta, np.inf, name="Update-or-Wait")
    return pol, static_cost(dist, beta, np.inf, kappa_s, kappa_p)


def optimized_static(dist: ServiceDist, kappa_s: float = 1.0, kappa_p: float = 1.0):
    """Best static (deadline, timeout) policy, found by exact cost minimisation."""
    f = lambda x: static_cost(dist, np.exp(x[0]), np.exp(x[1]), kappa_s, kappa_p)
    q = [float(dist.ppf(u)) for u in (0.5, 0.9, 0.99)]
    best = None
    for D0 in (0.1 * dist.mean, dist.mean, 3 * dist.mean):
        for T0 in q + [dist.mean]:
            r = optimize.minimize(f, x0=np.log([max(D0, 1e-6), max(T0, 1e-6)]), method="Nelder-Mead",
                                  options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    D, T = np.exp(best.x)
    # compare with never preempting
    pol_np, c_np = update_or_wait(dist, kappa_s, kappa_p, rule="cost")
    if c_np <= best.fun:
        return StaticPolicy(pol_np.D, np.inf, name="Optimized Static"), c_np
    return StaticPolicy(float(D), float(T), name="Optimized Static"), float(best.fun)
