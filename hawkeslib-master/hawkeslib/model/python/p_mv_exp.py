
import numpy as np

# use the pure-Python sampler you put in p_mv_sample.py
from .p_mv_sample import mv_sample_branching, DelayDistribution
TINY = np.finfo(float).tiny


def mv_exp_sample_branching(T: float,
                            mu: np.ndarray,
                            A: np.ndarray,
                            theta: float):
    """
    Backward-compatible wrapper to the generic sampler, exponential kernel case.
    Mirrors the old signature and forwards to mv_sample_branching with EXPONENTIAL.
    """
    mu = np.asarray(mu, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    # theta is the exponential rate; other thetas unused for EXPONENTIAL
    return mv_sample_branching(T, mu, A, theta, 0.0, 0.0, 0.0, DelayDistribution.EXPONENTIAL)


# def mv_exp_ll(t: np.ndarray,
#               c: np.ndarray,
#               mu: np.ndarray,
#               A: np.ndarray,
#               theta: float,
#               T: float) -> float:
#     """
#     Compute log likelihood for a multivariate Hawkes process with exponential decay.

#     Parameters
#     ----------
#     t : (N,) float64, strictly increasing event times in [0, T)
#     c : (N,) int64, marks in {0, …, K-1}
#     mu : (K,) float64, background intensities
#     A  : (K,K) float64, infectivity matrix
#     theta : float > 0, exponential decay rate
#     T : float > max(t), observation horizon
#     """
#     t = np.asarray(t, dtype=np.float64)
#     c = np.asarray(c, dtype=np.int64)
#     mu = np.asarray(mu, dtype=np.float64)
#     A  = np.asarray(A,  dtype=np.float64)

#     if t.size != c.size:
#         raise ValueError("t and c must have the same length.")
#     if t.size and (np.any(np.diff(t) < 0)):
#         raise ValueError("t must be nondecreasing (chronological order).")

#     N = t.shape[0]
#     if N == 0:
#         return -np.sum(mu) * T

#     K = mu.shape[0]

#     # State arrays
#     phi = np.zeros(K, dtype=np.float64)
#     d   = np.full(K, np.inf, dtype=np.float64)  # ages since last event for each process
#     ed  = np.zeros(K, dtype=np.float64)
#     F   = np.zeros(K, dtype=np.float64)

#     # first event
#     F[c[0]] += 1.0 - np.exp(-theta * (T - t[0]))
#     lJ = np.log(max(mu[c[0]], np.finfo(float).tiny))
#     d[c[0]] = 0.0

#     # remaining events
#     for i in range(1, N):
#         ci = int(c[i])
#         ti = float(t[i])

#         dt = ti - t[i - 1]
#         d += dt
#         np.multiply(-theta, d, out=ed)
#         np.exp(ed, out=ed)

#         Aphi = np.sum(A[:, ci] * ed * (1.0 + phi))
#         lda = mu[ci] + theta * Aphi
#         lJ += np.log(max(lda, np.finfo(float).tiny))

#         F[ci] += 1.0 - np.exp(-theta * (T - ti))

#         phi[ci] = ed[ci] * (1.0 + phi[ci])
#         d[ci] = 0.0

#     return lJ - np.sum(mu * T) - np.sum(A.T.dot(F))
def mv_exp_ll(t, c, mu, A, theta, T):
    t = np.asarray(t, float)
    c = np.asarray(c, int)
    mu = np.asarray(mu, float)
    A  = np.asarray(A,  float)

    if t.size != c.size:
        raise ValueError("t and c must have the same length.")
    if t.size and np.any(np.diff(t) < 0):
        raise ValueError("t must be nondecreasing.")
    if t.size and not (0 <= t.min() and t.max() < T):
        raise ValueError("All event times must be in [0, T).")

    K = mu.shape[0]
    N = t.size
    if N == 0:
        return -float(mu.sum()) * T

    # Boundary integral term per parent mark m:
    # F[m] = sum_{j: c_j=m} (1 - exp(-theta*(T - t_j)))
    F = np.zeros(K, float)
    for ti, ci in zip(t, c):
        F[ci] += 1.0 - np.exp(-theta * (T - ti))

    # Event contribution using S recursion
    S = np.zeros(K, float)  # S_m(t-) = sum_{j: c_j=m, t_j<t} exp(-theta*(t - t_j))
    last_t = 0.0
    logJ = 0.0

    for ti, ki in zip(t, c):
        dt = ti - last_t
        if dt > 0:
            S *= np.exp(-theta * dt)

        lam = mu[ki] + theta * (A[:, ki] @ S)
        logJ += np.log(lam if lam > TINY else TINY)

        S[ki] += 1.0
        last_t = ti

    return logJ - float(mu.sum()) * T - float((A.T @ F).sum())


# ---------- E-step using correct recursion ----------
def _e_step(t, c, mu, A, theta, T, K):
    """
    Returns E1 (K,), E2 (K,K), C1 (K,) under current params.
    - E1[k]  accumulates expected background events for child k.
    - E2[m,k] accumulates expected number of m->k triggered events.
    - C1[m]  = sum_{j: c_j=m} (1 - exp(-theta*(T - t_j))) for M-step of A.
    """
    E1 = np.zeros(K, float)
    E2 = np.zeros((K, K), float)
    C1 = np.zeros(K, float)

    S = np.zeros(K, float)
    last_t = 0.0

    for ti, ki in zip(t, c):
        dt = ti - last_t
        if dt > 0:
            S *= np.exp(-theta * dt)

        # responsibilities at this event (child = ki)
        denom = mu[ki] + theta * (A[:, ki] @ S)
        denom = denom if denom > TINY else TINY

        q0  = mu[ki] / denom                    # background
        qm  = theta * A[:, ki] * S / denom      # vector over parents m

        E1[ki]    += q0
        E2[:, ki] += qm

        # boundary integral bookkeeping for A row-normalization
        C1[ki] += 1.0 - np.exp(-theta * (T - ti))

        S[ki] += 1.0
        last_t = ti

    return E1, E2, C1


# ---------- 1-D GEM step for theta (golden-section on log-theta) ----------
def _maximize_theta(t, c, mu, A, T, theta0, br_lo=0.2, br_hi=5.0, iters=40):
    """
    Maximize LL over theta with mu, A held fixed.
    Search on log-theta in [log(theta0*br_lo), log(theta0*br_hi)].
    If theta0 is tiny, expands the lower bound.
    """
    theta0 = float(theta0)
    # guard bounds
    lo = max(theta0 * br_lo, 1e-12)
    hi = max(theta0 * br_hi, lo * 10.0)
    phi = (np.sqrt(5.0) - 1.0) / 2.0

    def f(logth):
        th = np.exp(logth)
        return mv_exp_ll(t, c, mu, A, th, T)

    a, b = np.log(lo), np.log(hi)
    c1 = b - phi * (b - a)
    c2 = a + phi * (b - a)
    f1 = f(c1)
    f2 = f(c2)

    for _ in range(iters):
        if f1 < f2:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + phi * (b - a)
            f2 = f(c2)
        else:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - phi * (b - a)
            f1 = f(c1)

    logth_star = (a + b) / 2.0
    theta_star = float(np.exp(logth_star))
    ll_star = f(logth_star)
    return theta_star, ll_star


# ---------- Main EM/GEM ----------
def mv_exp_fit_em(t, c, T, maxiter=200, reltol=1e-5, init=None, seed=None,
                  theta_search_lo=0.2, theta_search_hi=5.0, theta_search_iters=40):
    """
    EM for multivariate Hawkes with exponential kernel.
    - Closed-form M-step for μ and A (given θ).
    - GEM step: 1-D maximize observed LL over θ each iteration.

    Parameters
    ----------
    t : (N,) float, nondecreasing in [0, T)
    c : (N,) int, marks in {0,..,K-1} (contiguous integers)
    T : float, horizon > max(t)
    init : dict or None with optional keys {'mu','A','theta'}
    seed : int or None for RNG used only if init is None

    Returns
    -------
    odll : float
    (mu, A, theta) : parameters
    n_iter : int
    """
    t = np.asarray(t, float)
    c = np.asarray(c, int)
    if t.size == 0:
        raise ValueError("Empty sequence.")
    if t.size != c.size:
        raise ValueError("t and c must have same length.")
    if np.any(np.diff(t) < 0):
        raise ValueError("t must be nondecreasing.")
    if not (0 <= t.min() and t.max() < T):
        raise ValueError("All event times must lie in [0, T).")

    # require marks 0..K-1 contiguous
    uniq = np.unique(c)
    if not np.array_equal(uniq, np.arange(uniq.size)):
        # remap (safe, deterministic)
        remap = {u: i for i, u in enumerate(uniq)}
        c = np.vectorize(remap.get)(c).astype(int)

    K = uniq.size
    N = t.size

    # --- initialization ---
    if init is not None:
        mu = np.array(init.get('mu', np.full(K, N / (T * K))), float)
        A  = np.array(init.get('A',  np.eye(K) * 0.1), float)
        theta = float(init.get('theta', max(1.0 / (np.median(np.diff(t)[np.diff(t) > 0]) if N > 1 else T), 1e-6)))
    else:
        rng = np.random.default_rng(seed)
        rate_scale = max(N / (T * K), 1e-6)
        mu = rng.random(K) * rate_scale
        A  = np.full((K, K), 0.05)
        np.fill_diagonal(A, 0.1)
        theta = max(rate_scale * 0.5, 1e-6)

    mu = np.maximum(mu, 1e-12)
    A  = np.maximum(A, 0.0)
    theta = float(theta)

    odll_prev = -np.inf

    for it in range(1, maxiter + 1):
        # ---- E-step under current (mu, A, theta) ----
        E1, E2, C1 = _e_step(t, c, mu, A, theta, T, K)

        # ---- M-step for μ and A (given theta) ----
        mu_new = E1 / T
        mu_new = np.maximum(mu_new, 1e-12)

        C1_safe = C1.copy()
        C1_safe[C1_safe == 0.0] = np.inf
        A_new = E2 / C1_safe[:, None]
        A_new = np.maximum(A_new, 0.0)

        # ---- GEM step for θ with μ, A fixed ----
        theta_new, ll_theta = _maximize_theta(
            t, c, mu_new, A_new, T, theta,
            br_lo=theta_search_lo, br_hi=theta_search_hi, iters=theta_search_iters
        )

        # observed LL with updated (μ, A, θ)
        odll = mv_exp_ll(t, c, mu_new, A_new, theta_new, T)

        # monotonicity safety: accept only if not decreasing (within tiny tolerance)
        if np.isfinite(odll_prev):
            rel_imp = (odll - odll_prev) / (abs(odll_prev) if odll_prev != 0 else 1.0)
            if rel_imp < -1e-8:
                # backtrack θ only (keep μ,A updates, which are valid M-steps)
                odll_bt = mv_exp_ll(t, c, mu_new, A_new, theta, T)
                if odll_bt >= odll:  # accept backtracked
                    theta_new = theta
                    odll = odll_bt

                # if still worse, stop to avoid thrashing
                if odll < odll_prev:
                    return odll_prev, (mu, A, theta), it - 1

            if rel_imp < reltol:
                return odll, (mu_new, A_new, theta_new), it

        # commit
        mu, A, theta = mu_new, A_new, theta_new
        odll_prev = odll

    return odll_prev, (mu, A, theta), maxiter


# Beta constant
def mv_exp_fit_em_fix_theta(t, c, T, theta, maxiter=200, reltol=1e-5, init=None, seed=None,
                  theta_search_lo=0.2, theta_search_hi=5.0, theta_search_iters=40):
    """
    EM for multivariate Hawkes with exponential kernel.
    - Closed-form M-step for μ and A (given θ).
    - GEM step: 1-D maximize observed LL over θ each iteration.

    Parameters
    ----------
    t : (N,) float, nondecreasing in [0, T)
    c : (N,) int, marks in {0,..,K-1} (contiguous integers)
    T : float, horizon > max(t)
    init : dict or None with optional keys {'mu','A','theta'}
    seed : int or None for RNG used only if init is None

    Returns
    -------
    odll : float
    (mu, A, theta) : parameters
    n_iter : int
    """
    t = np.asarray(t, float)
    c = np.asarray(c, int)
    if t.size == 0:
        raise ValueError("Empty sequence.")
    if t.size != c.size:
        raise ValueError("t and c must have same length.")
    if np.any(np.diff(t) < 0):
        raise ValueError("t must be nondecreasing.")
    if not (0 <= t.min() and t.max() < T):
        raise ValueError("All event times must lie in [0, T).")

    # require marks 0..K-1 contiguous
    uniq = np.unique(c)
    if not np.array_equal(uniq, np.arange(uniq.size)):
        # remap (safe, deterministic)
        remap = {u: i for i, u in enumerate(uniq)}
        c = np.vectorize(remap.get)(c).astype(int)

    K = uniq.size
    N = t.size

    # --- initialization ---
    if init is not None:
        mu = np.array(init.get('mu', np.full(K, N / (T * K))), float)
        A  = np.array(init.get('A',  np.eye(K) * 0.1), float)
    else:
        rng = np.random.default_rng(seed)
        rate_scale = max(N / (T * K), 1e-6)
        mu = rng.random(K) * rate_scale
        A  = np.full((K, K), 0.05)
        np.fill_diagonal(A, 0.1)

    mu = np.maximum(mu, 1e-12)
    A  = np.maximum(A, 0.0)

    odll_prev = -np.inf

    for it in range(1, maxiter + 1):
        # ---- E-step under current (mu, A, theta) ----
        E1, E2, C1 = _e_step(t, c, mu, A, theta, T, K)

        # ---- M-step for μ and A (given theta) ----
        mu_new = E1 / T
        mu_new = np.maximum(mu_new, 1e-12)

        C1_safe = C1.copy()
        C1_safe[C1_safe == 0.0] = np.inf
        A_new = E2 / C1_safe[:, None]
        A_new = np.maximum(A_new, 0.0)

        # observed LL with updated (μ, A, θ)
        odll = mv_exp_ll(t, c, mu_new, A_new, theta, T)

        # monotonicity safety: accept only if not decreasing (within tiny tolerance)
        if np.isfinite(odll_prev):
            rel_imp = (odll - odll_prev) / (abs(odll_prev) if odll_prev != 0 else 1.0)
            if rel_imp < -1e-8:
                # backtrack θ only (keep μ,A updates, which are valid M-steps)
                odll_bt = mv_exp_ll(t, c, mu_new, A_new, theta, T)
                if odll_bt >= odll:  # accept backtracked
                    odll = odll_bt

                # if still worse, stop to avoid thrashing
                if odll < odll_prev:
                    return odll_prev, (mu, A, theta), it - 1

            if rel_imp < reltol:
                return odll, (mu_new, A_new, theta), it

        # commit
        mu, A  = mu_new, A_new
        odll_prev = odll

    return odll_prev, (mu, A, theta), maxiter
