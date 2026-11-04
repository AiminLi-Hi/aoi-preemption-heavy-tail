"""Command-line interface.

    python -m tailor_aoi --dist lognormal --mean 2 --sigma2 6 --ks 1 --kp 1
    python -m tailor_aoi --dist pareto2 --mean 2 --var 40 --ks 1 --kp 1
    python -m tailor_aoi --dist exponential --mean 1
"""

import argparse

from . import ServiceDist, cite, compare, solve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tailor_aoi", description="Age-optimal sampling and preemption (TAILOR).")
    ap.add_argument("--dist", choices=["lognormal", "pareto2", "exponential"], default="lognormal")
    ap.add_argument("--mean", type=float, default=2.0, help="mean service time E[Y]")
    ap.add_argument("--sigma2", type=float, default=4.0, help="log-normal: variance of ln Y")
    ap.add_argument("--var", type=float, default=40.0, help="Pareto II: variance of Y")
    ap.add_argument("--ks", type=float, default=1.0, help="sampling cost")
    ap.add_argument("--kp", type=float, default=1.0, help="preemption cost")
    ap.add_argument("--dt", type=float, default=0.005, help="grid step")
    ap.add_argument("--cite", action="store_true", help="print the BibTeX entry and exit")
    a = ap.parse_args(argv)
    if a.cite:
        print(cite())
        return
    if a.dist == "lognormal":
        d = ServiceDist.lognormal_from_mean(a.mean, a.sigma2)
    elif a.dist == "pareto2":
        d = ServiceDist.pareto2(a.mean, a.var)
    else:
        d = ServiceDist.exponential(1.0 / a.mean)
    p = solve(d, a.ks, a.kp, dt=a.dt)
    print(p.summary())
    print(f"  sample when AoI reaches z*(0) = {float(p.sampling_target(0.0)):.4f}")
    print(f"  preempt a fresh service after theta*(z*(0)) = {float(p.preemption_time(p.sampling_target(0.0))):.4f}")
    print(compare(d, a.ks, a.kp, policy=p))
    print("\nIf this is useful in your research, please cite:  python -m tailor_aoi --cite")


if __name__ == "__main__":
    main()
