"""Service-time distributions used by TAILOR.

A ``ServiceDist`` only needs a survival function, a density, a quantile
function and a sampler.  Partial moments ``E[Y^n; Y <= a]`` (n = 0, 1, 2) are
used by the exact baseline evaluators; closed forms are provided where they
exist and numerical quadrature is used otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, special, stats


@dataclass(frozen=True)
class ServiceDist:
    """A continuous, non-negative service-time distribution ``Y``.

    Use the constructors :meth:`lomax`, :meth:`pareto2`, :meth:`lognormal`,
    :meth:`exponential` or :meth:`from_scipy` instead of building it by hand.
    """

    name: str
    sf: Callable[[np.ndarray], np.ndarray]
    pdf: Callable[[np.ndarray], np.ndarray]
    ppf: Callable[[np.ndarray], np.ndarray]
    sampler: Callable[[np.random.Generator, int], np.ndarray]
    mean: float
    second_moment: float
    _partial: Callable[[int, float], float] | None = None

    # ------------------------------------------------------------------ API
    @property
    def variance(self) -> float:
        return self.second_moment - self.mean**2

    def cdf(self, t):
        return 1.0 - self.sf(t)

    def sample(self, n: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        return self.sampler(rng, int(n))

    def partial_moment(self, n: int, a: float) -> float:
        """Return ``E[Y^n ; Y <= a]`` for n in {0, 1, 2}."""
        if a <= 0:
            return 0.0
        if np.isinf(a):
            return [1.0, self.mean, self.second_moment][n]
        if self._partial is not None:
            return float(self._partial(n, a))
        val, _ = integrate.quad(lambda s: s**n * float(self.pdf(np.array(s))), 0.0, a, limit=400)
        return float(val)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ServiceDist({self.name}, mean={self.mean:.4g}, var={self.variance:.4g})"

    # --------------------------------------------------------- constructors
    @staticmethod
    def lomax(alpha: float, scale: float = 1.0) -> "ServiceDist":
        """Pareto type II (Lomax): ``P(Y > t) = (1 + t/scale)^(-alpha)``.

        Requires ``alpha > 2`` so that ``E[Y^2] < inf``.
        """
        if alpha <= 2:
            raise ValueError("Lomax needs alpha > 2 for a finite second moment.")
        k, a = float(scale), float(alpha)
        mean = k / (a - 1)
        m2 = 2 * k**2 / ((a - 1) * (a - 2))

        def partial(n, x):
            u = 1 + x / k
            if n == 0:
                return 1 - u ** (-a)
            # E[Y^n; Y<=x] = n * int_0^x s^{n-1} sf(s) ds - x^n sf(x)
            if n == 1:
                integral = k / (a - 1) * (1 - u ** (1 - a))
                return integral - x * u ** (-a)
            integral = k**2 * ((u ** (2 - a) - 1) / (2 - a) - (u ** (1 - a) - 1) / (1 - a))
            return 2 * integral - x**2 * u ** (-a)

        return ServiceDist(
            name=f"Lomax(alpha={a:g}, scale={k:g})",
            sf=lambda t: (1 + np.asarray(t, float) / k) ** (-a),
            pdf=lambda t: (a / k) * (1 + np.asarray(t, float) / k) ** (-a - 1),
            ppf=lambda p: k * ((1 - np.asarray(p, float)) ** (-1 / a) - 1),
            sampler=lambda rng, n: k * (rng.random(n) ** (-1 / a) - 1),
            mean=mean,
            second_moment=m2,
            _partial=partial,
        )

    @staticmethod
    def pareto2(mean: float, variance: float) -> "ServiceDist":
        """Lomax (Pareto II) parameterised by its mean and variance."""
        # var/mean^2 = alpha / (alpha - 2)  =>  alpha = 2 r / (r - 1), r = var/mean^2
        r = variance / mean**2
        if r <= 1:
            raise ValueError("Pareto II needs variance > mean^2.")
        alpha = 2 * r / (r - 1)
        return ServiceDist.lomax(alpha, scale=mean * (alpha - 1))

    @staticmethod
    def lognormal(mu: float, sigma: float) -> "ServiceDist":
        """Log-normal: ``ln Y ~ N(mu, sigma^2)``."""
        mu, s = float(mu), float(sigma)
        d = stats.lognorm(s=s, scale=np.exp(mu))

        def partial(n, x):
            z = (np.log(x) - mu - n * s**2) / s
            return np.exp(n * mu + 0.5 * n**2 * s**2) * special.ndtr(z)

        return ServiceDist(
            name=f"LogNormal(mu={mu:g}, sigma^2={s**2:g})",
            sf=lambda t: d.sf(np.asarray(t, float)),
            pdf=lambda t: d.pdf(np.asarray(t, float)),
            ppf=lambda p: d.ppf(np.asarray(p, float)),
            sampler=lambda rng, n: np.exp(mu + s * rng.standard_normal(n)),
            mean=float(np.exp(mu + s**2 / 2)),
            second_moment=float(np.exp(2 * mu + 2 * s**2)),
            _partial=partial,
        )

    @staticmethod
    def lognormal_from_mean(mean: float, sigma2: float) -> "ServiceDist":
        """Log-normal with a given mean and log-variance ``sigma^2`` (as in the paper)."""
        return ServiceDist.lognormal(np.log(mean) - sigma2 / 2, np.sqrt(sigma2))

    @staticmethod
    def exponential(rate: float) -> "ServiceDist":
        lam = float(rate)

        def partial(n, x):
            e = np.exp(-lam * x)
            if n == 0:
                return 1 - e
            if n == 1:
                return (1 - e * (1 + lam * x)) / lam
            return (2 - e * (2 + 2 * lam * x + (lam * x) ** 2)) / lam**2

        return ServiceDist(
            name=f"Exp(rate={lam:g})",
            sf=lambda t: np.exp(-lam * np.asarray(t, float)),
            pdf=lambda t: lam * np.exp(-lam * np.asarray(t, float)),
            ppf=lambda p: -np.log1p(-np.asarray(p, float)) / lam,
            sampler=lambda rng, n: rng.exponential(1 / lam, n),
            mean=1 / lam,
            second_moment=2 / lam**2,
            _partial=partial,
        )

    @staticmethod
    def from_scipy(frozen, name: str | None = None) -> "ServiceDist":
        """Wrap any frozen ``scipy.stats`` continuous distribution on ``[0, inf)``."""
        m1, m2 = frozen.mean(), frozen.moment(2)
        if not (np.isfinite(m1) and np.isfinite(m2)):
            raise ValueError("TAILOR needs E[Y] < inf and E[Y^2] < inf.")
        return ServiceDist(
            name=name or frozen.dist.name,
            sf=lambda t: frozen.sf(np.asarray(t, float)),
            pdf=lambda t: frozen.pdf(np.asarray(t, float)),
            ppf=lambda p: frozen.ppf(np.asarray(p, float)),
            sampler=lambda rng, n: frozen.rvs(size=n, random_state=rng),
            mean=float(m1),
            second_moment=float(m2),
        )
