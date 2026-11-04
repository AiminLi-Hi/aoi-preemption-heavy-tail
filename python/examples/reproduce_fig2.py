"""Reproduce Fig. 2 of the ISIT 2026 paper with exact baseline costs.

Pareto II and log-normal service times with E[Y] = 2 and k_s = k_p = 1.
TAILOR uses the ACOE optimum rho; the baselines use exact renewal-reward costs.

    python examples/reproduce_fig2.py            # writes fig2_tailor.png
"""

import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401  (keeps default fonts)

from tailor_aoi import ServiceDist, solve, update_or_wait, zero_wait

KS = KP = 1.0
MEAN = 2.0
pareto_var = [10, 20, 30, 40, 50, 60, 70, 80]
logn_s2 = [1, 2, 3, 4, 5, 6]


def sweep(dists):
    out = {"TAILOR": [], "UoW": [], "ZW": []}
    for d in dists:
        out["TAILOR"].append(solve(d, KS, KP).rho)
        out["UoW"].append(update_or_wait(d, KS, KP, rule="sun")[1])
        out["ZW"].append(zero_wait(d, KS, KP)[1])
    return out


def main(path="fig2_tailor.png"):
    par = sweep([ServiceDist.pareto2(MEAN, v) for v in pareto_var])
    logn = sweep([ServiceDist.lognormal_from_mean(MEAN, s2) for s2 in logn_s2])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    styles = [
        ("ZW", "Zero-Wait", dict(color="black", ls="--", marker="o", mfc="white")),
        ("UoW", "Update-or-Wait (Sun et al. 2017)", dict(color="#1f4fd1", ls=":", marker="s", mfc="white")),
        ("TAILOR", r"TAILOR ($\bf{Proposed}$)", dict(color="#e31a1c", ls="-", marker="D", lw=2.2)),
    ]
    for ax, x, res, xl, ttl in [
        (axes[0], pareto_var, par, r"Var$(Y)$", "a) Pareto II"),
        (axes[1], logn_s2, logn, r"$\sigma^2$", "b) Log-normal"),
    ]:
        for key, lab, st in styles:
            ax.plot(x, res[key], label=lab, ms=5, **st)
        ax.set_xlabel(xl)
        ax.set_title(ttl, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("average AoI + action cost")
    axes[1].set_yscale("log")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    for name, res, xs in [("Pareto II", par, pareto_var), ("Log-normal", logn, logn_s2)]:
        print(name)
        for i, xv in enumerate(xs):
            print(f"  x={xv:>3}: TAILOR {res['TAILOR'][i]:8.4f} | UoW {res['UoW'][i]:9.3f} | ZW {res['ZW'][i]:9.3f}")
    return path


if __name__ == "__main__":
    main()
