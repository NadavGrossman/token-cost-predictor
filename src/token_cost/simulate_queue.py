"""
Queue-scheduling simulation — does predicting output length buy shorter waits?

Model: M/G/1. Requests arrive as a Poisson process of rate λ; one non-preemptive
server (the "LLM") answers them one at a time. No model is run — service time is
a linear stand-in for autoregressive decoding (see config: overhead + tokens /
rate), drawn i.i.d. by sampling test prompts with replacement, so the service
distribution G is the empirical answer-length distribution. λ is set from a
target utilisation ρ = λ·E[S].

The queue is dynamic: whenever the server frees up it picks the best job among
those that have *arrived and are still waiting*, so the choice depends on the
state of the queue at that instant, not on a pre-sorted batch. Policies differ
only in the priority they sort that waiting set by:

    fcfs           arrival order
    sjf_input      shortest prompt first (naive baseline)
    sjf_predicted  strict non-preemptive priority by our model's predicted bucket
    sjf_oracle     shortest true service time first (upper bound)

Ties keep arrival order (FIFO within a priority class). All four are
work-conserving and non-preemptive, so they see the same workload process and
differ only in who waits.

Analytic cross-checks (same assumptions, closed form):
    Pollaczek–Khinchine   W_fcfs = λ·E[S²] / (2(1 − ρ))
    Cobham (1954)         W_k    = W₀ / ((1 − σ_{k−1})(1 − σ_k)),  W₀ = λ·E[S²]/2
    Kleinrock conservation Σ_k ρ_k·W_k is the same for every policy above

Usage:
    uv run python -m token_cost.simulate_queue --dataset wildchat
"""
import argparse
import heapq
import json

import numpy as np
import pandas as pd

from token_cost import config

# Lower key is served first; each returns the priority of every sampled job.
POLICIES = {
    "fcfs":          lambda job: np.arange(len(job["service"]), dtype=float),
    "sjf_input":     lambda job: job["prompt_tokens"].astype(float),
    "sjf_predicted": lambda job: job["pred_bucket"].astype(float),
    "sjf_oracle":    lambda job: job["service"],
}

RHOS = (0.50, 0.70, 0.85, 0.90, 0.95)
HEADLINE_RHO = 0.90


def service_seconds(output_tokens: np.ndarray) -> np.ndarray:
    return config.SERVICE_OVERHEAD_SEC + output_tokens / config.DECODE_TOKENS_PER_SEC


def sample_jobs(df: pd.DataFrame, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw n requests i.i.d. from the test set — the arrival stream's job mix."""
    idx = rng.integers(0, len(df), size=n)
    return {
        "service":       service_seconds(df["eval_count"].to_numpy(dtype=float)[idx]),
        "prompt_tokens": df["prompt_eval_count"].to_numpy()[idx],
        "pred_bucket":   df["pred_bucket"].to_numpy()[idx],
    }


def run_policy(arrival: np.ndarray, service: np.ndarray, priority: np.ndarray) -> np.ndarray:
    """Event-driven single non-preemptive server; returns each job's waiting time."""
    n = len(arrival)
    wait = np.empty(n)
    ready: list[tuple[float, int]] = []
    t = 0.0
    nxt = 0            # index of the next job still to arrive
    for _ in range(n):
        if not ready:
            t = max(t, arrival[nxt])
        while nxt < n and arrival[nxt] <= t:
            heapq.heappush(ready, (priority[nxt], nxt))   # index tie-break = FIFO
            nxt += 1
        _, j = heapq.heappop(ready)
        wait[j] = t - arrival[j]
        t += service[j]
    return wait


def n_priority_classes(df: pd.DataFrame, n_buckets: int | None) -> int:
    """Priority levels = predicted buckets. Default matches the four-way task."""
    k = config.N_BUCKETS if n_buckets is None else n_buckets
    if k < 2:
        raise ValueError(f"n_buckets must be >= 2, got {k}")
    hi = int(df["pred_bucket"].max()) + 1
    if hi > k:
        raise ValueError(f"pred_bucket goes up to {hi - 1} but n_buckets={k}")
    return k


def simulate(df: pd.DataFrame, rho: float, n_jobs: int, warmup: int, reps: int,
             rng: np.random.Generator, n_buckets: int | None = None) -> dict:
    """Replicated runs at one utilisation; every policy sees the same arrivals."""
    k = n_priority_classes(df, n_buckets)
    lam = rho / service_seconds(df["eval_count"].to_numpy(dtype=float)).mean()
    per_rep: dict[str, list[float]] = {p: [] for p in POLICIES}
    p95: dict[str, list[float]] = {p: [] for p in POLICIES}
    worst: dict[str, list[float]] = {p: [] for p in POLICIES}
    class_wait = np.zeros(k)
    class_n = np.zeros(k)

    for _ in range(reps):
        job = sample_jobs(df, n_jobs, rng)
        arrival = np.cumsum(rng.exponential(1.0 / lam, size=n_jobs))
        for name, key_fn in POLICIES.items():
            wait = run_policy(arrival, job["service"], key_fn(job))[warmup:]
            per_rep[name].append(float(wait.mean()))
            p95[name].append(float(np.percentile(wait, 95)))
            worst[name].append(float(wait.max()))
            if name == "sjf_predicted":
                bucket = job["pred_bucket"][warmup:]
                for c in range(k):
                    class_wait[c] += wait[bucket == c].sum()
                    class_n[c] += (bucket == c).sum()

    return {
        "lambda_per_sec": lam,
        "avg_wait_sec": {p: float(np.mean(v)) for p, v in per_rep.items()},
        "std_sec":      {p: float(np.std(v))  for p, v in per_rep.items()},
        "p95_wait_sec": {p: float(np.mean(v)) for p, v in p95.items()},
        "max_wait_sec": {p: float(np.mean(v)) for p, v in worst.items()},
        "predicted_class_wait_sec": (class_wait / np.maximum(class_n, 1)).tolist(),
        "theory": theory(df, lam, n_buckets=k),
    }


def theory(df: pd.DataFrame, lam: float, n_buckets: int | None = None) -> dict:
    """Closed-form M/G/1 waits: P–K for FCFS, Cobham for the bucket priority."""
    k = n_priority_classes(df, n_buckets)
    s = service_seconds(df["eval_count"].to_numpy(dtype=float))
    rho = lam * s.mean()
    residual = lam * (s ** 2).mean() / 2          # W₀, mean remaining work on arrival
    fcfs = residual / (1 - rho)

    bucket = df["pred_bucket"].to_numpy()
    share = np.array([(bucket == c).mean() for c in range(k)])
    load = np.array([lam * s[bucket == c].sum() / len(s) for c in range(k)])
    cum = np.concatenate([[0.0], np.cumsum(load)])
    per_class = residual / ((1 - cum[:-1]) * (1 - cum[1:]))
    return {
        "rho": float(rho),
        "mean_service_sec": float(s.mean()),
        "second_moment_sec2": float((s ** 2).mean()),
        "scv": float(s.var() / s.mean() ** 2),
        "fcfs_pk_sec": float(fcfs),
        "priority_cobham_sec": float((share * per_class).sum()),
        "priority_cobham_class_sec": per_class.tolist(),
        "class_share": share.tolist(),
        "class_load": load.tolist(),
        # Kleinrock's conservation law: identical for every work-conserving,
        # non-preemptive discipline. Equality with rho * fcfs validates Cobham.
        "conservation_priority": float((load * per_class).sum()),
        "conservation_fcfs": float(rho * fcfs),
    }


def print_table(title: str, res: dict) -> None:
    w, t = res["avg_wait_sec"], res["theory"]
    print(f"\n{title}   (λ = {res['lambda_per_sec']:.4f}/s, ρ = {t['rho']:.2f})")
    print(f"{'policy':<15}{'sim avg wait':>15}{'p95':>10}{'vs fcfs':>10}{'theory':>10}")
    print("-" * 60)
    closed = {"fcfs": t["fcfs_pk_sec"], "sjf_predicted": t["priority_cobham_sec"]}
    for name, v in w.items():
        saving = f"{(1 - v / w['fcfs']) * 100:>8.1f}%" if name != "fcfs" else f"{'—':>9}"
        th = f"{closed[name]:>9.1f}s" if name in closed else f"{'—':>10}"
        print(f"{name:<15}{v:>13.1f} s{res['p95_wait_sec'][name]:>10.1f}{saving}{th}")


def main(dataset: str, n_jobs: int, warmup: int, reps: int,
         encoder: str | None = None, max_seq_len: int | None = None,
         n_buckets: int | None = None) -> None:
    splits_dir = config.dataset_paths(dataset)["splits_dir"]
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    preds   = pd.read_parquet(config.predictions_file(
        dataset, "classification", encoder, max_seq_len, n_buckets))
    # predictions are row-aligned with the test split (ids are not unique)
    assert len(preds) == len(test_df) and (preds["id"] == test_df["id"]).all()
    df = test_df.assign(pred_bucket=preds["pred_bucket"].to_numpy())
    k = n_priority_classes(df, n_buckets)

    rng = np.random.default_rng(config.RANDOM_SEED)
    by_rho = {}
    for rho in RHOS:
        by_rho[f"{rho:.2f}"] = simulate(df, rho, n_jobs, warmup, reps, rng, n_buckets=k)
        print_table(f"Utilisation ρ = {rho:.2f}", by_rho[f"{rho:.2f}"])

    base = by_rho[f"{HEADLINE_RHO:.2f}"]["theory"]
    slug = config.run_slug(encoder, max_seq_len, n_buckets)
    out_path = (config.METRICS_DIR / f"queue_sim_{dataset}.json" if slug is None
                else config.METRICS_DIR / f"queue_sim_{dataset}_{slug}.json")
    out_path.write_text(json.dumps({
        "dataset": dataset,
        "encoder": encoder or config.DEFAULT_ENCODER,
        "n_buckets": k,
        "model": "M/G/1, Poisson arrivals, single non-preemptive server, dynamic priority",
        "service_time": {"overhead_sec": config.SERVICE_OVERHEAD_SEC,
                         "tokens_per_sec": config.DECODE_TOKENS_PER_SEC,
                         "mean_sec": base["mean_service_sec"],
                         "second_moment_sec2": base["second_moment_sec2"],
                         "scv": base["scv"]},
        "run": {"jobs_per_rep": n_jobs, "warmup_jobs": warmup, "reps": reps,
                "test_prompts": len(df)},
        "headline_rho": f"{HEADLINE_RHO:.2f}",
        "by_rho": by_rho,
    }, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wildchat", choices=list(config.DATASETS))
    parser.add_argument("--jobs", type=int, default=200_000, help="requests per replication")
    parser.add_argument("--warmup", type=int, default=20_000, help="requests discarded per replication")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--encoder", default=config.DEFAULT_ENCODER,
                        choices=list(config.ENCODERS))
    parser.add_argument("--n-buckets", type=int, default=None)
    args = parser.parse_args()
    main(dataset=args.dataset, n_jobs=args.jobs, warmup=args.warmup, reps=args.reps,
         encoder=args.encoder, n_buckets=args.n_buckets)
