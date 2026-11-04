"""tailor-aoi: age-optimal sampling and preemption under heavy-tailed delays.

Quick start::

    from tailor_aoi import ServiceDist, solve, compare

    dist = ServiceDist.lognormal_from_mean(mean=2.0, sigma2=4.0)
    policy = solve(dist, kappa_s=1.0, kappa_p=1.0)
    print(policy.summary())
    print(compare(dist, 1.0, 1.0, policy=policy))

Reference: A. Li, Y. Ince, E. Uysal, "Taming the Heavy Tail: Age-Optimal
Preemption", Proc. IEEE ISIT 2026. DOI: 10.1109/ISIT62367.2026.11653616
"""

from .baselines import StaticPolicy, optimized_static, static_cost, update_or_wait, zero_wait
from .distributions import ServiceDist
from .simulate import SimResult, sample_path, simulate
from .solver import Policy, solve

__version__ = "0.1.0"

CITATION = """@inproceedings{li2026taming,
  title     = {Taming the Heavy Tail: Age-Optimal Preemption},
  author    = {Li, Aimin and {\\.I}nce, Yi{\\u{g}}it and Uysal, Elif},
  booktitle = {Proc. IEEE International Symposium on Information Theory (ISIT)},
  year      = {2026},
  doi       = {10.1109/ISIT62367.2026.11653616}
}"""


def compare(dist, kappa_s=1.0, kappa_p=1.0, policy=None, **solve_kwargs):
    """Average cost of TAILOR vs. the baselines used in the paper.

    Baseline costs are exact (renewal-reward); TAILOR's cost is the optimal
    average cost ``rho`` returned by the ACOE solver.
    """
    if policy is None:
        policy = solve(dist, kappa_s, kappa_p, **solve_kwargs)
    rows = [("TAILOR (Proposed)", policy.rho)]
    rows.append(("Update-or-Wait [Sun et al. 2017]", update_or_wait(dist, kappa_s, kappa_p, rule="sun")[1]))
    rows.append(("Zero-Wait", zero_wait(dist, kappa_s, kappa_p)[1]))
    rows.append(("Optimized Static (D, T)", optimized_static(dist, kappa_s, kappa_p)[1]))
    width = max(len(r[0]) for r in rows)
    lines = [f"{dist.name}, k_s={kappa_s:g}, k_p={kappa_p:g}"]
    lines += [f"  {n:<{width}}  {c:12.5f}   ({c / policy.rho:6.2f}x)" for n, c in rows]
    return "\n".join(lines)


def cite() -> str:
    """BibTeX entry to cite when you use this package."""
    return CITATION


__all__ = [
    "ServiceDist", "Policy", "solve", "StaticPolicy", "static_cost", "zero_wait",
    "update_or_wait", "optimized_static", "simulate", "sample_path", "SimResult",
    "compare", "cite", "CITATION",
]
