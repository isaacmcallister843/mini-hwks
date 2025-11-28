# c_mv_samp_py.py
"""
Generic sampling code (and special cases) for multivariate Hawkes processes
with factorized kernels — pure Python/Numpy port.
"""

from enum import IntEnum
import numpy as np


class DelayDistribution(IntEnum):
    EXPONENTIAL = 0
    BETA = 1


def _get_mv_offspring_exp(t: float,
                          Acp: np.ndarray,
                          theta: float,
                          T: float):
    """
    Offspring for exponential delay.
    Parameters
    ----------
    t : float
        Parent time.
    Acp : (K,) array
        Row A[c_parent, :].
    theta : float
        Exponential rate (>0).
    T : float
        Observation horizon.

    Returns
    -------
    tos : (M,) float64 array
    cos : (M,) int64 array
        Only times < T are returned.
    """
    Acp = np.asarray(Acp, dtype=np.float64)
    K = Acp.shape[0]

    tos_list = []
    cos_list = []

    # For each child-dimension k, draw N_k ~ Poisson(Acp[k]) children
    for k in range(K):
        lam = Acp[k]
        if lam <= 0:
            continue
        Nk = np.random.poisson(lam)
        if Nk == 0:
            continue
        # Exponential with mean 1/theta; add parent time
        tt = t + np.random.exponential(scale=1.0 / theta, size=Nk)
        tos_list.append(tt)
        cos_list.append(np.full(Nk, k, dtype=np.int64))

    if tos_list:
        tos = np.concatenate(tos_list).astype(np.float64, copy=False)
        cos = np.concatenate(cos_list).astype(np.int64, copy=False)
        m = tos < T
        return tos[m], cos[m]
    else:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)


def _get_mv_offspring_generic(t: float,
                              Acp: np.ndarray,
                              theta1: float, theta2: float, theta3: float, theta4: float,
                              T: float,
                              distid: int | DelayDistribution):
    """
    Generic offspring generator supporting EXPONENTIAL or BETA delays.

    For BETA, we interpret:
        theta1 = alpha, theta2 = beta, theta3 = tmax (scale to [0, tmax]).

    theta4 is unused (kept for signature compatibility).
    """
    Acp = np.asarray(Acp, dtype=np.float64)
    K = Acp.shape[0]

    # Accept raw ints (0/1) for backward compatibility
    try:
        distid = DelayDistribution(int(distid))
    except Exception:
        raise ValueError("distid must be 0/1 or a DelayDistribution enum.")

    tos_list = []
    cos_list = []

    for k in range(K):
        lam = Acp[k]
        if lam <= 0:
            continue
        Nk = np.random.poisson(lam)
        if Nk == 0:
            continue

        if distid is DelayDistribution.EXPONENTIAL:
            tt = t + np.random.exponential(scale=1.0 / theta1, size=Nk)
        elif distid is DelayDistribution.BETA:
            # sample in [0, theta3]
            tt = t + np.random.beta(theta1, theta2, size=Nk) * theta3
        else:
            raise ValueError(f"Unsupported DelayDistribution: {distid}")

        tos_list.append(tt)
        cos_list.append(np.full(Nk, k, dtype=np.int64))

    if tos_list:
        tos = np.concatenate(tos_list).astype(np.float64, copy=False)
        cos = np.concatenate(cos_list).astype(np.int64, copy=False)
        m = tos < T
        return tos[m], cos[m]
    else:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)


def mv_sample_branching(T: float,
                        mu: np.ndarray,
                        A: np.ndarray,
                        theta1: float, theta2: float, theta3: float, theta4: float,
                        distid: int | DelayDistribution):
    """
    Generic branching sampler for multivariate Hawkes with
    factorized & normalized triggering kernels.

    Parameters
    ----------
    T : float
        Observation horizon.
    mu : (K,) array
        Background intensities.
    A : (K, K) array
        Infectivity matrix (each A[i, j] is mean number of children in j per parent in i).
    theta1..theta4 : floats
        Kernel parameters (meaning depends on `distid`).
    distid : {0,1} or DelayDistribution
        0/EXPONENTIAL or 1/BETA.

    Returns
    -------
    P : (N,) float64
        Sorted event times in [0, T).
    C : (N,) int64
        Corresponding marks in {0, …, K-1}.
    """
    mu = np.asarray(mu, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    K = mu.shape[0]

    try:
        distid = DelayDistribution(int(distid))
    except Exception:
        raise ValueError("distid must be 0/1 or a DelayDistribution enum.")

    # Immigrants
    P = []
    C = []

    for k in range(K):
        rate = mu[k] * T
        if rate <= 0:
            continue
        Nk0 = np.random.poisson(rate)
        if Nk0 == 0:
            continue
        Pk0 = np.random.random(Nk0) * T
        Ck0 = np.full(Nk0, k, dtype=np.int64)
        P.append(Pk0)
        C.append(Ck0)

    if P:
        curr_P = np.concatenate(P).astype(np.float64, copy=False)
        curr_C = np.concatenate(C).astype(np.int64, copy=False)
    else:
        # no immigrants → empty sample
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    # Clear accumulators; we'll rebuild as we go
    P = []
    C = []

    # Branching (cluster) expansion
    while curr_P.size > 0:
        # accumulate current generation
        P.append(curr_P)
        C.append(curr_C)

        os_P_list = []
        os_C_list = []

        # generate offspring per parent
        for i in range(curr_P.size):
            ci = int(curr_C[i])
            ti = float(curr_P[i])

            if distid is DelayDistribution.EXPONENTIAL:
                tres, cres = _get_mv_offspring_exp(ti, A[ci, :], theta1, T)
            elif distid is DelayDistribution.BETA:
                tres, cres = _get_mv_offspring_generic(
                    ti, A[ci, :], theta1, theta2, theta3, 0.0, T, DelayDistribution.BETA
                )
            else:
                raise ValueError(f"Unsupported DelayDistribution: {distid}")

            if tres.size:
                os_P_list.append(tres)
                os_C_list.append(cres)

        if os_P_list:
            curr_P = np.concatenate(os_P_list).astype(np.float64, copy=False)
            curr_C = np.concatenate(os_C_list).astype(np.int64, copy=False)
        else:
            # no more offspring; terminate
            break

    if P:
        P = np.concatenate(P).astype(np.float64, copy=False)
        C = np.concatenate(C).astype(np.int64, copy=False)
    else:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    # Stable sort by time to match mergesort behavior from the Cython version
    six = np.argsort(P, kind="mergesort")
    return P[six], C[six]
