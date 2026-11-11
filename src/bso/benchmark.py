"""Benchmark: Bayesian optimization vs. random search vs. Latin hypercube.

    python -m bso.benchmark --seeds 20 --budget 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .optimize import bayes_opt, lhs_search, make_gp, random_search
from .simulator import ReactorSimulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def run(seeds: int, budget: int, noise: float) -> dict:
    sim = ReactorSimulator()
    x_opt, y_opt = sim.true_optimum()
    curves = {"Bayesian optimization (GP + EI)": [], "Latin hypercube": [], "Random search": []}
    for s in range(seeds):
        f = ReactorSimulator(noise_sd=noise, seed=s)
        clean = ReactorSimulator()
        hist, _ = bayes_opt(f, f.bounds, budget=budget, seed=s)
        # score each method by the TRUE value of its best-observed point (noise-free)
        for name, h in [("Bayesian optimization (GP + EI)", hist),
                        ("Latin hypercube", lhs_search(f, f.bounds, budget, s)),
                        ("Random search", random_search(f, f.bounds, budget, s))]:
            true_y = clean(clean.bounds.from_unit(np.array(h.X)))
            best_idx = [int(np.argmax(np.array(h.y)[: i + 1])) for i in range(len(h.y))]
            curves[name].append([float(y_opt - true_y[j]) for j in best_idx])
    summary = {}
    for name, runs in curves.items():
        r = np.array(runs)
        final = r[:, -1]
        hit = [(next((i + 1 for i, v in enumerate(run) if v <= 0.01 * abs(y_opt)), None)) for run in runs]
        summary[name] = {
            "median_regret_final": float(np.median(final)),
            "p90_regret_final": float(np.quantile(final, 0.9)),
            "share_within_1pct": float(np.mean(final <= 0.01 * abs(y_opt))),
            # runs that never get within 1% count as "more than the budget"
            "median_evals_to_1pct": (lambda m: None if not np.isfinite(m) else float(m))(
                float(np.median([h if h else np.inf for h in hit]))),
            "curve_median": np.median(r, axis=0).tolist(),
            "curve_q25": np.quantile(r, 0.25, axis=0).tolist(),
            "curve_q75": np.quantile(r, 0.75, axis=0).tolist(),
        }
    return {"optimum_x": x_opt.tolist(), "optimum_y": y_opt, "seeds": seeds, "budget": budget,
            "noise_sd": noise, "methods": summary}


def surrogate_quality(budget: int = 30, seed: int = 0) -> dict:
    """How well does the GP fitted on BO samples vs. space-filling samples predict the landscape?"""
    sim = ReactorSimulator()
    g = np.linspace(0, 1, 60)
    U = np.array([[a, b] for a in g for b in g])
    Y = sim(sim.bounds.from_unit(U))
    out = {}
    hist, _ = bayes_opt(sim, sim.bounds, budget=budget, seed=seed)
    designs = {"BO samples": np.array(hist.X), "Latin hypercube": np.array(lhs_search(sim, sim.bounds, budget, seed).X)}
    for name, X in designs.items():
        gp = make_gp(seed).fit(X, sim(sim.bounds.from_unit(X)))
        mu, sd = gp.predict(U, return_std=True)
        top = Y >= np.quantile(Y, 0.9)  # the region we actually care about
        out[name] = {
            "rmse_global": float(np.sqrt(np.mean((mu - Y) ** 2))),
            "rmse_top10pct": float(np.sqrt(np.mean((mu[top] - Y[top]) ** 2))),
            "coverage_95": float(np.mean(np.abs(mu - Y) <= 1.96 * sd)),
        }
    return out


def plot(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colors = {"Bayesian optimization (GP + EI)": "#2b6cb0", "Latin hypercube": "#dd6b20", "Random search": "#718096"}
    for name, s in result["methods"].items():
        x = np.arange(1, len(s["curve_median"]) + 1)
        ax.plot(x, s["curve_median"], label=name, color=colors[name], lw=2)
        ax.fill_between(x, s["curve_q25"], s["curve_q75"], color=colors[name], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("Simulator evaluations")
    ax.set_ylabel("Regret (optimum − best found)")
    ax.set_title(f"Median regret over {result['seeds']} seeds (band = IQR)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def plot_landscape(path: Path, seed: int = 0, budget: int = 30) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = ReactorSimulator()
    a = np.linspace(*[sim.bounds.low[0], sim.bounds.high[0]], 200)
    b = np.linspace(*[sim.bounds.low[1], sim.bounds.high[1]], 200)
    A, B = np.meshgrid(a, b)
    Z = sim.objective(A, B)
    hist, _ = bayes_opt(sim, sim.bounds, budget=budget, seed=seed)
    P = sim.bounds.from_unit(np.array(hist.X))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cs = ax.contourf(A, B, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, label="Operating margin")
    ax.scatter(P[:6, 0], P[:6, 1], c="white", edgecolor="k", s=40, label="initial design (6)")
    ax.scatter(P[6:, 0], P[6:, 1], c="#e53e3e", edgecolor="k", s=40, label="BO proposals")
    xo, _ = sim.true_optimum()
    ax.scatter([xo[0]], [xo[1]], marker="*", c="gold", edgecolor="k", s=220, label="true optimum")
    ax.set_xlabel("Residence time (min)")
    ax.set_ylabel("Coolant temperature (°C)")
    ax.set_title("Where Bayesian optimization spends its 30 evaluations")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--noise", type=float, default=0.0)
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    res = run(args.seeds, args.budget, args.noise)
    res["surrogate_quality"] = surrogate_quality(args.budget)
    (REPORTS / "benchmark.json").write_text(json.dumps(res, indent=2))
    plot(res, REPORTS / "regret.png")
    plot_landscape(REPORTS / "landscape.png", budget=args.budget)
    print(f"true optimum: x={np.round(res['optimum_x'], 2).tolist()}, margin={res['optimum_y']:.4f}")
    print("| Method | Median final regret | 90th pct regret | Runs within 1% of optimum | Median evals to reach 1% |")
    print("|---|---|---|---|---|")
    for name, s in res["methods"].items():
        e = s["median_evals_to_1pct"]
        print(f"| {name} | {s['median_regret_final']:.4f} | {s['p90_regret_final']:.4f} | "
              f"{s['share_within_1pct']:.0%} | {f'> {res["budget"]}' if e is None else int(e)} |")
    print("\n| Surrogate trained on | RMSE (whole space) | RMSE (top 10% region) | 95% interval coverage |")
    print("|---|---|---|---|")
    for name, q in res["surrogate_quality"].items():
        print(f"| {name} | {q['rmse_global']:.4f} | {q['rmse_top10pct']:.4f} | {q['coverage_95']:.0%} |")


if __name__ == "__main__":
    main()
