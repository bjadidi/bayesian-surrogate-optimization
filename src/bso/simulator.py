"""Stand-in for an expensive process simulation.

`ReactorSimulator` is an analytic surrogate of a continuous stirred-tank reactor
(first-order exothermic reaction A -> B with a competing side reaction) solved at
steady state. Inputs are residence time and coolant temperature; the output is based on the
yield of B. It is cheap to evaluate here, but we *pretend* each call costs hours (as
a CFD or DEM run would), so the goal is to find the optimum in as few calls as
possible.

The objective is an operating margin: yield minus a cost per minute of residence
time (longer runs mean lower throughput). That puts the optimum on a tilted, elongated ridge
in the interior, next to a thermal-runaway cliff, which is what makes naive grid
or random search wasteful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R = 8.314  # J/mol/K


@dataclass(frozen=True)
class Bounds:
    names: tuple[str, ...] = ("residence_time_min", "coolant_temp_C")
    low: tuple[float, ...] = (1.0, 20.0)
    high: tuple[float, ...] = (60.0, 120.0)

    def as_array(self) -> np.ndarray:
        return np.array([self.low, self.high], dtype=float).T

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        lo, hi = np.array(self.low), np.array(self.high)
        return (np.asarray(x, dtype=float) - lo) / (hi - lo)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        lo, hi = np.array(self.low), np.array(self.high)
        return lo + np.asarray(u, dtype=float) * (hi - lo)


class ReactorSimulator:
    """Operating margin of a CSTR with A -> B (desired) and A -> C (side reaction)."""

    bounds = Bounds()
    cost_per_min = 0.006  # margin lost per minute of residence time (throughput cost)

    def __init__(self, noise_sd: float = 0.0, seed: int | None = None) -> None:
        self.noise_sd = noise_sd
        self.rng = np.random.default_rng(seed)
        self.n_calls = 0

    @staticmethod
    def _yield(tau_min: np.ndarray, t_cool_c: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_min, dtype=float)
        t = np.asarray(t_cool_c, dtype=float) + 273.15
        # reactor runs hotter than the coolant; heat release grows with conversion and residence time
        t_r = t + 18.0 * (1 - np.exp(-tau / 25.0))
        k1 = 4.0e6 * np.exp(-55_000 / (R * t_r))      # desired, lower activation energy
        k2 = 6.0e12 * np.exp(-100_000 / (R * t_r))    # side reaction, takes over when hot
        k = k1 + k2
        conversion = k * tau / (1 + k * tau)
        selectivity = k1 / k
        y = conversion * selectivity
        # thermal runaway penalty above ~395 K reactor temperature
        runaway = 1 / (1 + np.exp(-(t_r - 395.0) / 3.0))
        return y * (1 - 0.85 * runaway)

    @classmethod
    def objective(cls, tau_min, t_cool_c):
        return cls._yield(tau_min, t_cool_c) - cls.cost_per_min * np.asarray(tau_min, dtype=float)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        self.n_calls += len(x)
        y = self.objective(x[:, 0], x[:, 1])
        if self.noise_sd:
            y = y + self.rng.normal(0, self.noise_sd, size=y.shape)
        return y

    def true_optimum(self, grid: int = 400) -> tuple[np.ndarray, float]:
        """Dense grid search, used only to score the optimizers (never by them)."""
        a = np.linspace(self.bounds.low[0], self.bounds.high[0], grid)
        b = np.linspace(self.bounds.low[1], self.bounds.high[1], grid)
        A, B = np.meshgrid(a, b)
        Y = self.objective(A, B)
        i = np.unravel_index(np.argmax(Y), Y.shape)
        return np.array([A[i], B[i]]), float(Y[i])
