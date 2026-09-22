"""
Heteroscedastic-Gaussian target utilities (shared by train / evaluate / baselines).

Output-token count is heavy-tailed, so we model it as **log-normal**: the target
is ``y = log1p(count)`` and we assume ``y ~ Normal(mu, sigma)``. The network
predicts ``mu`` and ``log sigma`` in a *standardized* space (zero-mean / unit-std
``y``) for stable optimization; everything here converts between the three spaces:

    token count  <--log1p/expm1-->  log space (y)  <--standardize-->  model space (z)

A single prompt therefore yields, in token space, a predicted median plus an
(asymmetric, never-negative) central interval for any coverage level.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

ArrayLike = np.ndarray | list[float]


# ── token count  <->  log space ───────────────────────────────────────────────
def to_log(counts: ArrayLike) -> np.ndarray:
    return np.log1p(np.asarray(counts, dtype=float))


def to_counts(log_vals: ArrayLike) -> np.ndarray:
    return np.expm1(np.asarray(log_vals, dtype=float))


# ── standardization stats (computed on train, persisted per dataset) ──────────
def compute_stats(counts: ArrayLike) -> dict[str, float]:
    """Mean/std of the log-space target; std floored to avoid divide-by-zero."""
    y = to_log(counts)
    return {"mean": float(y.mean()), "std": float(y.std()) or 1.0}


def standardize(y_log: ArrayLike, stats: dict[str, float]) -> np.ndarray:
    return (np.asarray(y_log, dtype=float) - stats["mean"]) / stats["std"]


# ── model output (mu_z, log_sigma_z)  ->  log-space params (mu, sigma) ─────────
def model_to_log_params(
    mu_z: ArrayLike, log_sigma_z: ArrayLike, stats: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """De-standardize a (mean, log-std) prediction back into log space."""
    mu_z = np.asarray(mu_z, dtype=float)
    sigma_z = np.exp(np.asarray(log_sigma_z, dtype=float))
    mu_log = mu_z * stats["std"] + stats["mean"]
    sigma_log = sigma_z * stats["std"]
    return mu_log, sigma_log


# ── log-space params  ->  token-space, human-facing summary ───────────────────
def z_for_level(level: float) -> float:
    """Two-sided z-multiplier for a central coverage level (0.80 -> 1.2816)."""
    return NormalDist().inv_cdf(0.5 * (1.0 + level))


def predict_summary(
    mu_log: float, sigma_log: float, coverage_levels: tuple[float, ...]
) -> dict:
    """Per-prompt cost forecast in token space (median + central intervals)."""
    summary = {
        "median_tokens": float(np.expm1(mu_log)),
        "mean_tokens": float(np.expm1(mu_log + 0.5 * sigma_log**2)),
        "mu_log": float(mu_log),
        "sigma_log": float(sigma_log),
        "intervals": {},
    }
    for level in coverage_levels:
        z = z_for_level(level)
        summary["intervals"][f"{level:.2f}"] = {
            "low": float(np.expm1(mu_log - z * sigma_log)),
            "high": float(np.expm1(mu_log + z * sigma_log)),
        }
    return summary


# ── metrics (operate on whole arrays; shared by model + baselines) ────────────
def gaussian_nll(y_log: np.ndarray, mu_log: np.ndarray, sigma_log: np.ndarray) -> float:
    """Mean negative log-likelihood of the true log-targets under N(mu, sigma)."""
    sigma_log = np.maximum(sigma_log, 1e-6)
    nll = 0.5 * np.log(2 * np.pi) + np.log(sigma_log) + 0.5 * ((y_log - mu_log) / sigma_log) ** 2
    return float(nll.mean())


def coverage(y_log: np.ndarray, mu_log: np.ndarray, sigma_log: np.ndarray, level: float) -> float:
    """Empirical fraction of truths inside the predicted central interval."""
    z = z_for_level(level)
    sigma_log = np.maximum(sigma_log, 1e-6)
    inside = np.abs((y_log - mu_log) / sigma_log) <= z
    return float(inside.mean())


def mean_interval_width(mu_log: np.ndarray, sigma_log: np.ndarray, level: float) -> float:
    """Average token-space width of the predicted central interval."""
    z = z_for_level(level)
    low = np.expm1(mu_log - z * sigma_log)
    high = np.expm1(mu_log + z * sigma_log)
    return float((high - low).mean())
