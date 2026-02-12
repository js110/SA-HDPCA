from __future__ import annotations



from dataclasses import dataclass, field

from typing import List



import numpy as np





def static_schedule(eps_iter: float, T: int) -> np.ndarray:

    return np.full(T, eps_iter / T, dtype=float)





@dataclass

class FeedbackBudget:

    eps_iter: float

    T: int

    beta: float = 0.8

    gamma: float = 1.0

    eps_min_ratio: float = 0.2

    eps_max_ratio: float = 2.0

    warmup: int = 3

    drift_clip: float = 2.0

    raw_eps: List[float] = field(default_factory=list)

    m: float = 0.0

    last_drift_norm: float = 0.0

    t: int = 0



    def __post_init__(self):

        eps_avg = self.eps_iter / max(self.T, 1)

        self.eps_min = self.eps_min_ratio * eps_avg

        self.eps_max = self.eps_max_ratio * eps_avg



    def next_eps(self) -> tuple[float, float]:

        drift_input = self.last_drift_norm if self.t > 0 else 0.0

        self.m = self.beta * self.m + (1.0 - self.beta) * drift_input

        if self.t < self.warmup:

            eps_raw = self.eps_min

        else:

            gain = 1.0 / (1.0 + self.m / max(self.gamma, 1e-6))

            eps_raw = self.eps_min + (self.eps_max - self.eps_min) * gain

        self.raw_eps.append(eps_raw)

        self.t += 1

        return eps_raw, eps_raw



    def register_drift(self, drift: float, centroid_norm: float) -> None:

        denom = centroid_norm + 1e-6

        drift_norm = drift / denom if denom > 0 else 0.0

        drift_norm = float(np.clip(drift_norm, 0.0, self.drift_clip))

        self.last_drift_norm = drift_norm



    def finalize(self) -> list[float]:

        total = sum(self.raw_eps) + 1e-12

        scale = self.eps_iter / total

        return [e * scale for e in self.raw_eps]





@dataclass

class FeedbackBudgetV2:

    eps_iter: float

    T: int

    warmup: int = 3

    beta: float = 0.8

    gamma: float = 1.0

    drift_clip: float = 2.0

    eps_min_ratio: float = 0.2

    eps_max_ratio: float = 2.0

    p: float = 2.0

    adj_clip_low: float = 0.7

    adj_clip_high: float = 1.3

    recovery_scale: float = 0.5

    raw_eps: List[float] = field(default_factory=list)

    eps_base_trace: List[float] = field(default_factory=list)

    adj_trace: List[float] = field(default_factory=list)

    m: float = 0.0

    last_drift_norm: float = 0.0

    t: int = 0



    def __post_init__(self):

        eps_avg = self.eps_iter / max(self.T, 1)

        self.eps_min = self.eps_min_ratio * eps_avg

        self.eps_max = self.eps_max_ratio * eps_avg

        denom = max(self.T - 1, 1)

        self.base_schedule = [

            self.eps_min + (self.eps_max - self.eps_min) * ((t / denom) ** self.p) for t in range(self.T)

        ]



    def next_eps(self) -> tuple[float, float]:

        drift_input = self.last_drift_norm if self.t > 0 else 0.0

        self.m = self.beta * self.m + (1.0 - self.beta) * drift_input

        adj = 1.0 / (1.0 + self.m / max(self.gamma, 1e-6))

        adj = float(np.clip(adj, self.adj_clip_low, self.adj_clip_high))



        base = self.base_schedule[self.t]

        eps_raw = base * adj

        if self.t < self.warmup:

            eps_raw = self.eps_min

        self.eps_base_trace.append(base)

        self.adj_trace.append(adj)

        self.raw_eps.append(eps_raw)

        self.t += 1

        return eps_raw, eps_raw



    def register_drift(

        self,

        drift: float,

        centroid_norm: float | None = None,

        non_empty_k: float | None = None,

        max_cluster_ratio: float | None = None,

        k: int | None = None,

    ) -> None:

        denom = (centroid_norm if centroid_norm is not None else 0.0) + 1e-6

        drift_norm = drift / denom if denom > 0 else 0.0

        collapse_boost = 0.0

        if non_empty_k is not None and k:

            missing = max(0.0, (k - non_empty_k) / max(k, 1))

            collapse_boost += missing

        if max_cluster_ratio is not None:

            collapse_boost += max(0.0, max_cluster_ratio - 0.6)

        drift_norm = drift_norm - self.recovery_scale * collapse_boost

        drift_norm = float(np.clip(drift_norm, -self.drift_clip, self.drift_clip))

        self.last_drift_norm = drift_norm



    def finalize(self) -> list[float]:

        total = sum(self.raw_eps) + 1e-12

        scale = self.eps_iter / total

        return [e * scale for e in self.raw_eps]





def _feasible_bounds(remaining_budget: float, remaining_iters: int, eps_min: float, eps_max: float) -> tuple[float, float]:

    if remaining_iters <= 0:

        raise ValueError("remaining_iters must be positive.")

    lo = remaining_budget - (remaining_iters - 1) * eps_max

    hi = remaining_budget - (remaining_iters - 1) * eps_min

    lo = max(eps_min, lo)

    hi = min(eps_max, hi)

    return float(lo), float(hi)





@dataclass

class FeedbackBudgetV3:

    eps_iter: float

    T: int

    warmup: int = 3

    beta: float = 0.8

    gamma: float = 1.0

    drift_clip: float = 3.0

    eps_min_ratio: float = 0.6

    eps_max_ratio: float = 3.0

    p: float = 2.5

    time_power: float = 3.0

    adj_clip_low: float = 1.0

    adj_clip_high: float = 2.0

    recovery_scale: float = 1.0

    raw_eps: List[float] = field(default_factory=list)

    eps_base_trace: List[float] = field(default_factory=list)

    adj_trace: List[float] = field(default_factory=list)

    g_trace: List[float] = field(default_factory=list)

    m: float = 0.0

    last_drift_norm: float = 0.0

    t: int = 0

    remaining_budget: float = 0.0

    eps_avg: float = 0.0



    def __post_init__(self):

        if self.T <= 0:

            raise ValueError("T must be positive.")

        if self.eps_iter <= 0:

            raise ValueError("eps_iter must be positive.")

        eps_avg = self.eps_iter / float(self.T)

        self.eps_avg = float(eps_avg)

        self.eps_min = float(self.eps_min_ratio * eps_avg)

        self.eps_max = float(self.eps_max_ratio * eps_avg)

        if not (0 < self.eps_min <= self.eps_max):

            raise ValueError("Invalid eps_min/eps_max derived from ratios.")

        self.warmup = int(max(self.warmup, 0))

        self.warmup = min(self.warmup, self.T)

        denom = max(self.T - 1, 1)

        base = [self.eps_min + (self.eps_max - self.eps_min) * ((t / denom) ** float(self.p)) for t in range(self.T)]

        if self.time_power and self.T > self.warmup:

            tail_len = self.T - self.warmup

            u = np.linspace(0.0, 1.0, tail_len)

            bias = np.power(u + 1e-6, float(self.time_power))

            bias = bias / (bias.mean() + 1e-12)

            tail = np.array(base[self.warmup:], dtype=float) * bias

            base = base[: self.warmup] + tail.tolist()

        self.base_schedule = base

        self.remaining_budget = float(self.eps_iter)



    def next_eps(self) -> tuple[float, float]:

        if self.t >= self.T:

            raise ValueError("next_eps called more than T times.")

        drift_input = self.last_drift_norm if self.t > 0 else 0.0

        self.m = self.beta * self.m + (1.0 - self.beta) * drift_input

        g = 1.0 + self.m / max(self.gamma, 1e-6)

        g = float(np.clip(g, self.adj_clip_low, self.adj_clip_high))

        base = float(self.base_schedule[self.t])

        eps_raw = base * g

        if self.t < self.warmup:

            eps_raw = max(self.eps_min, self.eps_avg)



        eps_use = float(np.clip(eps_raw, self.eps_min, self.eps_max))



        self.eps_base_trace.append(base)

        self.adj_trace.append(g)

        self.g_trace.append(g)

        self.raw_eps.append(eps_use)

        self.t += 1

        return eps_use, eps_raw



    def register_drift(

        self,

        drift: float,

        centroid_norm: float,

        non_empty_k: float | None = None,

        max_cluster_ratio: float | None = None,

        k: int | None = None,

    ) -> None:

        denom = centroid_norm + 1e-6

        drift_norm = drift / denom if denom > 0 else 0.0

        collapse_boost = 0.0

        if non_empty_k is not None and k:

            missing = max(0.0, (k - non_empty_k) / max(k, 1))

            collapse_boost += missing

        if max_cluster_ratio is not None:

            collapse_boost += max(0.0, max_cluster_ratio - 0.6)

        drift_norm = drift_norm + self.recovery_scale * collapse_boost

        drift_norm = float(np.clip(drift_norm, 0.0, self.drift_clip))

        self.last_drift_norm = drift_norm



    def finalize(self) -> list[float]:

        total = sum(self.raw_eps) + 1e-12

        scale = self.eps_iter / total

        return [e * scale for e in self.raw_eps]

