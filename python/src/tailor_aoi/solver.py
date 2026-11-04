"""TAILOR: average-cost policy iteration for joint sampling and preemption.

This is a Python port of the MATLAB solver in ``Paretov2.m`` and implements
the coupled integral ACOEs of Theorem 1 in

    A. Li, Y. Ince, E. Uysal, "Taming the Heavy Tail: Age-Optimal Preemption",
    Proc. IEEE ISIT 2026.

Idle phase   h_I(D) = rho*D - D^2/2 + inf_{z>=D} { k_s + v(z) + z^2/2 - rho*z }
Busy phase   v(y)   = inf_{theta>=0} Q(y, theta; rho)
             Q      = (y - rho) A(theta) + J1(theta) + int_0^theta h_I dF
                      + sf(theta) * (k_p + v(y + theta))

Numerics follow Section V-A of the paper: a uniform grid in y, a hybrid
(uniform + log-spaced) action grid for theta, and a far-field linear closure
v(y) ~ v(y_cut) + s (y - y_cut) with slope s = E[min(Y, theta_max)].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .distributions import ServiceDist


def _cumtrapz(t: np.ndarray, g: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g, dtype=float)
    out[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(t))
    return out


@dataclass
class Policy:
    """Stationary TAILOR policy on the busy-start grid.

    * ``sampling_target(delta)`` returns z*(delta): wait until the AoI reaches
      this value, then sample (Theorem 1a).
    * ``preemption_time(y)`` returns theta*(y): starting a service with AoI y,
      preempt if it has not finished after this long (Theorem 1b).
    """

    rho: float
    y: np.ndarray
    v: np.ndarray
    z_map: np.ndarray
    theta_map: np.ndarray
    dt: float
    y_cut: float
    slope: float
    kappa_s: float
    kappa_p: float
    dist_name: str
    iterations: int
    converged: bool
    rho_history: list = field(default_factory=list)
    kind: str = "TAILOR"

    def sampling_target(self, delta):
        delta = np.asarray(delta, float)
        idx = np.rint(delta / self.dt).astype(int)
        inside = idx < self.z_map.size
        out = np.maximum(delta, self.rho - self.slope)  # far-field rule
        out = np.where(inside, self.z_map[np.minimum(idx, self.z_map.size - 1)], out)
        return out

    def wait_time(self, delta):
        return np.maximum(0.0, self.sampling_target(delta) - np.asarray(delta, float))

    def preemption_time(self, y):
        idx = np.rint(np.asarray(y, float) / self.dt).astype(int)
        return self.theta_map[np.minimum(idx, self.theta_map.size - 1)]

    def summary(self) -> str:
        return (
            f"{self.kind} policy for {self.dist_name}: rho = {self.rho:.6f} "
            f"(k_s={self.kappa_s:g}, k_p={self.kappa_p:g}; "
            f"{'converged' if self.converged else 'NOT converged'} in {self.iterations} iterations)"
        )


def _action_grid(dt, theta_cap, theta_fine, n_log):
    theta_fine = min(theta_fine, theta_cap)
    dense = np.arange(0.0, theta_fine + 0.5 * dt, dt)
    if theta_fine < theta_cap:
        a = max(theta_fine, dt)
        tail = np.exp(np.linspace(np.log(a), np.log(theta_cap), n_log))
        cand = np.unique(np.concatenate([dense, tail, [theta_cap]]))
    else:
        cand = dense
    cand = np.unique(np.floor(cand / dt + 1e-9) * dt)
    idx = np.unique(np.rint(cand / dt).astype(int))
    return idx[idx >= 1]  # forbid theta = 0


def solve(
    dist: ServiceDist,
    kappa_s: float = 1.0,
    kappa_p: float = 1.0,
    *,
    dt: float = 0.005,
    y_cut: float | None = None,
    eps_tail: float = 1e-6,
    theta_fine: float = 8.0,
    n_log: int = 250,
    max_iter: int = 60,
    tol_v: float = 1e-8,
    tol_rho: float = 1e-10,
    verbose: bool = False,
) -> Policy:
    """Compute the TAILOR (ACOE-optimal) sampling and preemption policy.

    Parameters
    ----------
    dist : ServiceDist
        Service-time distribution (needs E[Y^2] < inf).
    kappa_s, kappa_p : float
        Sampling and preemption costs.
    dt : float
        Grid step for the busy-start AoI ``y`` and the action ``theta``.
    y_cut : float, optional
        Truncation of the ``y`` grid.  Defaults to ``max(30, 15 E[Y])``.
    eps_tail : float
        Largest action ``theta_max`` satisfies ``P(Y > theta_max) = eps_tail``.
    theta_fine, n_log : float, int
        Uniform action grid on ``[0, theta_fine]`` and ``n_log`` log-spaced
        actions on ``[theta_fine, theta_max]``.
    """
    if y_cut is None:
        y_cut = max(30.0, 15.0 * dist.mean)
    y_cut = np.floor(y_cut / dt) * dt

    theta_cap = float(dist.ppf(1.0 - eps_tail))
    theta_cap = np.floor(theta_cap / dt) * dt
    y = np.arange(0.0, y_cut + 0.5 * dt, dt)
    M = y.size
    t = np.arange(0.0, theta_cap + 0.5 * dt, dt)
    T = t.size

    # Kernels: sf, density, A(theta)=int sf, J1(theta)=int s*sf(s) ds
    barF = np.asarray(dist.sf(t), float)
    f = np.asarray(dist.pdf(t), float)
    f[~np.isfinite(f)] = 0.0
    A = _cumtrapz(t, barF)
    J1 = _cumtrapz(t, t * barF)
    slope = float(A[-1])  # far-field slope  ~ E[min(Y, theta_max)] ~ E[Y]

    cand = _action_grid(dt, theta_cap, theta_fine, n_log)
    rho = 0.0
    v = np.zeros(M)
    tau_idx = np.full(M, cand[0])
    rho_hist = []
    converged = False
    z_map = y.copy()

    for it in range(1, max_iter + 1):
        v_old, rho_old, tau_old = v.copy(), rho, tau_idx.copy()
        v_cut = v[-1]

        # ---- Step 1: idle envelope  m(i) = min_{j>=i} phi(j)
        phi = kappa_s + v + 0.5 * y**2 - rho * y
        rev = phi[::-1]
        run_min = np.minimum.accumulate(rev)
        # argmin of the suffix (take the smallest index attaining the min, as in MATLAB)
        is_new = np.empty(M, bool)
        is_new[0] = True
        is_new[1:] = rev[1:] <= run_min[:-1]
        pos = np.where(is_new, np.arange(M), 0)
        pos = np.maximum.accumulate(pos)
        m_grid = run_min[::-1]
        arg_grid = (M - 1 - pos)[::-1]
        z_map = y[arg_grid]

        # ---- Step 2: h_I(t) on the action grid t
        const_ext = kappa_s + v_cut - slope * y_cut
        inside = t <= y_cut + 1e-12
        i0 = np.minimum(np.rint(t / dt).astype(int), M - 1)
        m1 = np.where(inside, m_grid[i0], np.inf)
        z1 = np.where(inside, y[arg_grid[i0]], np.inf)
        L = np.maximum(t, y_cut)
        z2 = np.maximum(L, rho - slope)
        m2 = 0.5 * z2**2 + (slope - rho) * z2 + const_ext
        use1 = m1 <= m2
        z_choice = np.where(use1, z1, z2)
        m_star = np.where(use1, m1, m2)
        hI = rho * t - 0.5 * t**2 + m_star
        I_fh = _cumtrapz(t, f * hI)

        # ---- Step 3: busy improvement (vectorised over y and candidate theta)
        tc = t[cand]
        base = J1[cand] - rho * A[cand] + I_fh[cand] + barF[cand] * kappa_p
        tau_idx = np.empty(M, dtype=int)
        for s in range(0, M, 512):  # chunk over y to bound memory
            ys = y[s:s + 512]
            yplus = ys[:, None] + tc[None, :]
            v_shift = np.where(
                yplus <= y_cut,
                np.interp(np.minimum(yplus, y_cut), y, v),
                v_cut + slope * (yplus - y_cut),
            )
            Q = ys[:, None] * A[cand] + base + barF[cand] * v_shift
            tau_idx[s:s + 512] = cand[np.argmin(Q, axis=1)]

        # ---- Step 4: policy evaluation (sparse linear system in (rho, v))
        w = z_choice - t
        Bw = _cumtrapz(t, f * w)
        Ccost = _cumtrapz(t, f * (t * w + 0.5 * w**2 + kappa_s))
        rho, v = _evaluate(
            y, t, dt, y_cut, slope, f, barF, A, J1, Bw, Ccost, z_choice, tau_idx, kappa_p
        )
        rho_hist.append(rho)

        dv = np.max(np.abs(v - v_old))
        dr = abs(rho - rho_old)
        dp = int(np.count_nonzero(tau_idx != tau_old))
        if verbose:
            print(f"iter {it:2d}: rho={rho:.8f}  |dv|={dv:.2e}  |drho|={dr:.2e}  policy changes={dp}")
        if dv <= tol_v and dr <= tol_rho and dp == 0:
            converged = True
            break

    return Policy(
        rho=float(rho),
        y=y,
        v=v,
        z_map=z_map,
        theta_map=t[tau_idx],
        dt=dt,
        y_cut=float(y_cut),
        slope=slope,
        kappa_s=kappa_s,
        kappa_p=kappa_p,
        dist_name=dist.name,
        iterations=it,
        converged=converged,
        rho_history=rho_hist,
    )


def _interp_weights(x, dt, M, y_cut, slope):
    """Linear-interpolation stencil of v at points x (with far-field closure)."""
    inside = x <= y_cut + 1e-12
    iL = np.clip(np.floor(x / dt + 1e-12).astype(int), 0, M - 2)
    lam = np.where(inside, x / dt - iL, 0.0)
    lam = np.clip(lam, 0.0, 1.0)
    iL = np.where(inside, iL, M - 1)
    const = np.where(inside, 0.0, slope * (x - y_cut))
    return iL, lam, inside, const


def _evaluate(y, t, dt, y_cut, slope, f, barF, A, J1, Bw, Ccost, z_choice, tau_idx, kappa_p):
    """Solve the Poisson equation for a fixed policy.

    The integral term int_0^theta f(t) v(z(t)) dt is handled with auxiliary
    cumulative unknowns S_j (one per distinct theta used by the policy) that
    are chained in increasing order of theta, which keeps the system sparse.
    """
    M = y.size
    used = np.unique(tau_idx)
    U = used.size
    n = 1 + M + U  # [rho, v_0..v_{M-1}, S_1..S_U]
    col_v = 1
    col_S = 1 + M

    # trapezoid weights: cumulative integral up to index j is
    #   sum_{k<=j} dt f_k g_k - dt/2 (f_0 g_0 + f_j g_j)
    iL, lam, inside, zconst = _interp_weights(z_choice, dt, M, y_cut, slope)
    wk = dt * f

    rows, cols, vals = [], [], []
    rhs = np.zeros(n)

    # Auxiliary chain: S_u - S_{u-1} - sum_{k in (prev, j]} wk * interp_k(v) = sum wk*const
    prev = -1
    for a, j in enumerate(used):
        r = col_S + a
        rows.append(r); cols.append(col_S + a); vals.append(1.0)
        if a > 0:
            rows.append(r); cols.append(col_S + a - 1); vals.append(-1.0)
        seg = slice(prev + 1, j + 1)
        w_seg = wk[seg]
        il, lm, ins = iL[seg], lam[seg], inside[seg]
        cw = np.bincount(il, weights=w_seg * np.where(ins, 1 - lm, 1.0), minlength=M)
        cw += np.bincount(np.minimum(il + 1, M - 1), weights=w_seg * np.where(ins, lm, 0.0), minlength=M)
        nz = np.nonzero(cw)[0]
        rows.extend([r] * nz.size); cols.extend((col_v + nz).tolist()); vals.extend((-cw[nz]).tolist())
        rhs[r] = float(np.sum(w_seg * zconst[seg]))
        prev = j
    # Bellman rows i = 0..M-2 (row M-1 replaced by normalisation v(0) = 0)
    a_of = {j: a for a, j in enumerate(used)}
    for i in range(M - 1):
        j = tau_idx[i]
        r = col_v + i
        g = A[j] + Bw[j]
        rv = y[i] * A[j] + J1[j] + Ccost[j] + barF[j] * kappa_p
        rows += [r, r]; cols += [0, col_v + i]; vals += [g, 1.0]
        # - c_j . v  =  - (S_j - dt/2 f_0 interp_0(v) - dt/2 f_j interp_j(v))
        rows.append(r); cols.append(col_S + a_of[j]); vals.append(-1.0)
        cst = 0.0
        for k in (0, j):
            hw = 0.5 * dt * f[k]
            if hw == 0.0:
                continue
            if inside[k]:
                rows += [r, r]; cols += [col_v + iL[k], col_v + min(iL[k] + 1, M - 1)]
                vals += [hw * (1 - lam[k]), hw * lam[k]]
            else:
                rows.append(r); cols.append(col_v + M - 1); vals.append(hw)
                cst -= hw * zconst[k]
        # S_j carries + sum wk*const, so c_j.v-const part: rhs gets + (S const) handled in chain;
        # remove the half-weights' constants at the endpoints.
        # - sf(theta) v(y_i + theta)
        yp = y[i] + t[j]
        if yp <= y_cut + 1e-12:
            il = min(int(np.floor(yp / dt + 1e-12)), M - 2)
            lm = yp / dt - il
            rows += [r, r]; cols += [col_v + il, col_v + il + 1]
            vals += [-barF[j] * (1 - lm), -barF[j] * lm]
            extra = 0.0
        else:
            rows.append(r); cols.append(col_v + M - 1); vals.append(-barF[j])
            extra = barF[j] * slope * (yp - y_cut)
        rhs[r] = rv + extra + cst
    # normalisation v(0) = 0  (replaces Bellman row at y_cut)
    r = col_v + M - 1
    rows.append(r); cols.append(col_v + 0); vals.append(1.0)
    # far-field slope closure v(M-1) - v(M-2) = slope * dt (uses the rho "row")
    rows += [0, 0]; cols += [col_v + M - 1, col_v + M - 2]; vals += [1.0, -1.0]
    rhs[0] = slope * dt

    Amat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    sol = spsolve(Amat.tocsc(), rhs)
    return float(sol[0]), sol[col_v:col_v + M]
