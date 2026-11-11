"""Gaussian-process Bayesian optimization and the baselines it is compared against."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def make_gp(seed: int = 0) -> GaussianProcessRegressor:
    """Matern-5/2 GP on unit-scaled inputs with a learned noise term."""
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=[0.2, 0.2], length_scale_bounds=(1e-3, 10.0), nu=2.5)
              + WhiteKernel(1e-5, (1e-9, 1e-1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5,
                                    random_state=seed)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


@dataclass
class History:
    X: list = field(default_factory=list)   # unit-scaled inputs
    y: list = field(default_factory=list)

    def best_so_far(self) -> np.ndarray:
        return np.maximum.accumulate(np.array(self.y))


def latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    return qmc.LatinHypercube(d=dim, seed=seed).random(n)


def bayes_opt(f, bounds, budget: int = 30, n_init: int = 6, seed: int = 0, xi: float = 0.01) -> tuple[History, GaussianProcessRegressor]:
    """Maximize f over bounds. f takes inputs in physical units."""
    rng = np.random.default_rng(seed)
    dim = len(bounds.low)
    hist = History()
    for u in latin_hypercube(n_init, dim, seed):
        hist.X.append(u)
        hist.y.append(float(f(bounds.from_unit(u))[0]))
    gp = make_gp(seed)
    while len(hist.y) < budget:
        X, y = np.array(hist.X), np.array(hist.y)
        gp.fit(X, y)
        best = y.max()
        cand = rng.random((4000, dim))
        mu, sd = gp.predict(cand, return_std=True)
        ei = expected_improvement(mu, sd, best, xi)
        starts = cand[np.argsort(ei)[-5:]]

        def neg_ei(u):
            m, s = gp.predict(u.reshape(1, -1), return_std=True)
            return -expected_improvement(m, s, best, xi)[0]

        results = [minimize(neg_ei, s0, bounds=[(0, 1)] * dim, method="L-BFGS-B") for s0 in starts]
        u_next = min(results, key=lambda r: r.fun).x
        if np.min(np.linalg.norm(X - u_next, axis=1)) < 1e-4:  # avoid re-sampling the same point
            u_next = cand[np.argmax(ei)]
        hist.X.append(u_next)
        hist.y.append(float(f(bounds.from_unit(u_next))[0]))
    gp.fit(np.array(hist.X), np.array(hist.y))
    return hist, gp


def random_search(f, bounds, budget: int = 30, seed: int = 0) -> History:
    rng = np.random.default_rng(seed)
    hist = History()
    for u in rng.random((budget, len(bounds.low))):
        hist.X.append(u)
        hist.y.append(float(f(bounds.from_unit(u))[0]))
    return hist


def lhs_search(f, bounds, budget: int = 30, seed: int = 0) -> History:
    hist = History()
    for u in latin_hypercube(budget, len(bounds.low), seed):
        hist.X.append(u)
        hist.y.append(float(f(bounds.from_unit(u))[0]))
    return hist
