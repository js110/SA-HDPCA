import numpy as np





def make_strong_contrast_schedule(

    eps_tot: float,

    T: int,

    low_frac: float = 0.7,

    eps_min_ratio: float = 0.7,

) -> list[float]:

    """
    Strong-contrast schedule for evidence/ablation:
    - First `low_frac` of iterations use eps_min_ratio * (eps_tot/T)
    - Remaining iterations share the remaining budget equally
    Sum is enforced to be exactly eps_tot (up to numerical precision).
    """

    if T <= 0:

        raise ValueError("T must be positive.")

    if eps_tot <= 0:

        raise ValueError("eps_tot must be positive.")

    low_frac = float(np.clip(low_frac, 0.0, 1.0))

    n_low = int(round(low_frac * T))

    n_low = min(max(n_low, 0), T)

    eps_avg = eps_tot / float(T)

    eps_min = float(eps_min_ratio) * eps_avg

    if n_low == 0:

        schedule = np.full(T, eps_avg, dtype=float)

        schedule[-1] += eps_tot - float(schedule.sum())

        return schedule.tolist()

    if n_low == T:

        schedule = np.full(T, eps_tot / float(T), dtype=float)

        schedule[-1] += eps_tot - float(schedule.sum())

        return schedule.tolist()



    remaining = eps_tot - n_low * eps_min

    if remaining <= 0:

        eps_min = eps_tot / float(T)

        schedule = np.full(T, eps_min, dtype=float)

        schedule[-1] += eps_tot - float(schedule.sum())

        return schedule.tolist()



    eps_high = remaining / float(T - n_low)

    schedule = np.concatenate([np.full(n_low, eps_min, dtype=float), np.full(T - n_low, eps_high, dtype=float)])

    schedule[-1] += eps_tot - float(schedule.sum())

    if np.any(schedule <= 0):

        raise ValueError("Non-positive eps in strong schedule; adjust eps_min_ratio/low_frac.")

    return schedule.tolist()





def make_fixed_tail_schedule(

    eps_tot: float,

    T: int,

    low_mult: float = 1.0,

    high_mult: float = 3.0,

) -> list[float]:

    """
    Drift-free, strictly increasing schedule:
      - weights linearly span [low_mult, high_mult]
      - normalized to sum to eps_tot
    """

    if T <= 0 or eps_tot <= 0:

        raise ValueError("Invalid T or eps_tot.")

    weights = np.linspace(low_mult, high_mult, T, dtype=float)

    weights = np.clip(weights, 1e-6, None)

    sched = weights / (weights.sum() + 1e-12) * eps_tot

    diff = eps_tot - float(sched.sum())

    sched[-1] += diff

    sched = np.clip(sched, 1e-6, None)

    return sched.tolist()

