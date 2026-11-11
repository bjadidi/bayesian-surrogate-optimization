import numpy as np

from bso.optimize import bayes_opt, expected_improvement, random_search
from bso.simulator import ReactorSimulator


def test_optimum_is_interior():
    x, y = ReactorSimulator().true_optimum(grid=200)
    b = ReactorSimulator.bounds
    assert b.low[0] < x[0] < b.high[0] and b.low[1] < x[1] < b.high[1]
    assert 0 < y < 1


def test_unit_scaling_round_trip():
    b = ReactorSimulator.bounds
    x = np.array([[12.0, 75.0], [60.0, 20.0]])
    assert np.allclose(b.from_unit(b.to_unit(x)), x)


def test_expected_improvement_properties():
    mu = np.array([0.0, 0.5, 1.0])
    sd = np.array([0.1, 0.1, 0.1])
    ei = expected_improvement(mu, sd, best=0.5, xi=0.0)
    assert np.all(ei >= 0) and ei[2] > ei[1] > ei[0]
    # more uncertainty -> more expected improvement at the same mean
    assert expected_improvement(np.array([0.4]), np.array([0.3]), 0.5)[0] > \
        expected_improvement(np.array([0.4]), np.array([0.05]), 0.5)[0]


def test_bayes_opt_beats_random_search_on_average():
    _, y_opt = ReactorSimulator().true_optimum(grid=200)
    bo, rs = [], []
    for s in range(3):
        f = ReactorSimulator()
        h, _ = bayes_opt(f, f.bounds, budget=20, n_init=6, seed=s)
        bo.append(y_opt - max(h.y))
        rs.append(y_opt - max(random_search(f, f.bounds, budget=20, seed=s).y))
    assert np.mean(bo) < np.mean(rs)
    assert np.mean(bo) < 0.01


def test_simulator_counts_calls_and_noise_is_seeded():
    a, b = ReactorSimulator(noise_sd=0.01, seed=1), ReactorSimulator(noise_sd=0.01, seed=1)
    x = np.array([[20.0, 90.0]])
    assert a(x)[0] == b(x)[0] and a.n_calls == 1
