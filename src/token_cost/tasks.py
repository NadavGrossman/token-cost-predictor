"""
Predictor tasks — pick the modeling approach with one ``--mode`` flag.

Two tasks share the *entire* pipeline (sampling, labeling, splits, the DistilBERT
backbone and training loop) and differ only in the small, well-defined surface
collected here:

  classification  4 length buckets, cross-entropy        -> accuracy / macro-F1
  regression      heteroscedastic Gaussian (mu, log std) -> NLL / calibrated range

Everything mode-specific (head size, per-row target, loss, validation score,
test metrics, baselines) lives behind the ``Task`` interface so the scripts stay
mode-agnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from token_cost import config
from token_cost import targets

LOG_SIGMA_CLAMP = config.LOG_SIGMA_CLAMP


# ── artifact persistence (bucket edges or target stats; both plain JSON) ───────
def save_artifact(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_artifact(path: Path) -> dict:
    return json.loads(Path(path).read_text())


class Task:
    """Interface every predictor mode implements."""

    name: str
    num_outputs: int
    artifact_filename: str
    target_dtype: torch.dtype
    summary_cols: list[str]

    def prepare(self, train_df: pd.DataFrame) -> dict:
        """Compute the per-dataset artifact (saved next to the splits)."""
        raise NotImplementedError

    def make_targets(self, df: pd.DataFrame, artifact: dict) -> np.ndarray:
        """Per-row training target."""
        raise NotImplementedError

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def val_score(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Validation score, *lower is better* (used to pick the best epoch)."""
        raise NotImplementedError

    def evaluate(self, train_df, test_df, logits: np.ndarray, artifact: dict) -> dict[str, dict]:
        """Test-set metrics for the model and every baseline."""
        raise NotImplementedError


# ── classification: length buckets ────────────────────────────────────────────
class ClassificationTask(Task):
    name = "classification"
    artifact_filename = "bucket_edges.json"
    target_dtype = torch.long
    summary_cols = ["accuracy", "macro_f1", "off_by_one"]

    def __init__(self, n_buckets: int | None = None) -> None:
        self.n_buckets = n_buckets or config.N_BUCKETS
        self.num_outputs = self.n_buckets
        self.artifact_filename = (
            "bucket_edges.json" if self.n_buckets == 4 else f"bucket_edges_k{self.n_buckets}.json"
        )

    def prepare(self, train_df: pd.DataFrame) -> dict:
        counts = train_df["eval_count"]
        steps = [i / self.n_buckets for i in range(1, self.n_buckets)]
        edges = sorted({int(np.quantile(counts, q)) for q in steps})
        if len(edges) < self.n_buckets - 1:   # degenerate (many ties): even spacing
            lo, hi = int(counts.min()), int(counts.max())
            step = max(1, (hi - lo) // self.n_buckets)
            edges = [lo + step * i for i in range(1, self.n_buckets)]
        return {"edges": edges, "n_buckets": self.n_buckets}

    @staticmethod
    def _to_bucket(counts, edges: list[int]) -> np.ndarray:
        return pd.cut(
            counts, bins=[-1] + edges + [float("inf")], labels=list(range(len(edges) + 1))
        ).astype(int).to_numpy()

    def make_targets(self, df: pd.DataFrame, artifact: dict) -> np.ndarray:
        return self._to_bucket(df["eval_count"], artifact["edges"])

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)

    def val_score(self, logits: np.ndarray, targets: np.ndarray) -> float:
        preds = logits.argmax(axis=1)
        return 1.0 - f1_score(targets, preds, average="macro", zero_division=0)

    def _metrics(self, true_b: np.ndarray, pred_b: np.ndarray) -> dict:
        n = len(true_b)
        return {
            "accuracy":   round(float((pred_b == true_b).mean()), 4),
            "macro_f1":   round(float(f1_score(true_b, pred_b, average="macro", zero_division=0)), 4),
            "off_by_one": round(float((np.abs(pred_b - true_b) <= 1).mean()), 4),
        }

    def evaluate(self, train_df, test_df, logits, artifact) -> dict[str, dict]:
        edges    = artifact["edges"]
        true_b   = self.make_targets(test_df, artifact)
        model_b  = logits.argmax(axis=1)

        majority = int(pd.Series(self.make_targets(train_df, artifact)).mode()[0])
        maj_b    = np.full(len(test_df), majority)
        inp_b    = self._to_bucket(test_df["prompt_eval_count"], edges)

        return {
            "distilbert":   self._metrics(true_b, model_b),
            "majority":     self._metrics(true_b, maj_b),
            "input_length": self._metrics(true_b, inp_b),
        }


# ── regression: heteroscedastic Gaussian ──────────────────────────────────────
class RegressionTask(Task):
    name = "regression"
    num_outputs = 2
    artifact_filename = "target_stats.json"
    target_dtype = torch.float

    def __init__(self) -> None:
        self.summary_cols = ["nll", "log_rmse", "token_mae", f"cov@{config.COVERAGE_LEVELS[0]:.2f}"]

    def prepare(self, train_df: pd.DataFrame) -> dict:
        return targets.compute_stats(train_df["eval_count"])

    def make_targets(self, df: pd.DataFrame, artifact: dict) -> np.ndarray:
        return targets.standardize(targets.to_log(df["eval_count"]), artifact)

    @staticmethod
    def _split(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return logits[:, 0], logits[:, 1].clamp(*LOG_SIGMA_CLAMP)

    def compute_loss(self, logits: torch.Tensor, targets_t: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self._split(logits)
        inv_var = torch.exp(-2.0 * log_sigma)
        nll = 0.5 * inv_var * (targets_t - mu) ** 2 + log_sigma
        if config.BETA_NLL > 0:   # beta-NLL: undo NLL's gradient down-weighting of hard points
            nll = nll * torch.exp(2.0 * log_sigma * config.BETA_NLL).detach()
        return nll.mean()

    def val_score(self, logits: np.ndarray, targets_z: np.ndarray) -> float:
        mu = logits[:, 0]
        log_sigma = np.clip(logits[:, 1], *LOG_SIGMA_CLAMP)
        inv_var = np.exp(-2.0 * log_sigma)
        return float((0.5 * inv_var * (targets_z - mu) ** 2 + log_sigma).mean())

    def _metrics(self, counts_true: np.ndarray, mu_log: np.ndarray, sigma_log: np.ndarray) -> dict:
        y_true = targets.to_log(counts_true)
        median = targets.to_counts(mu_log)
        m = {
            "nll":        round(targets.gaussian_nll(y_true, mu_log, sigma_log), 4),
            "log_mae":    round(float(np.abs(y_true - mu_log).mean()), 4),
            "log_rmse":   round(float(np.sqrt(((y_true - mu_log) ** 2).mean())), 4),
            "token_mae":  round(float(np.abs(counts_true - median).mean()), 2),
            "token_rmse": round(float(np.sqrt(((counts_true - median) ** 2).mean())), 2),
        }
        for level in config.COVERAGE_LEVELS:
            m[f"cov@{level:.2f}"]   = round(targets.coverage(y_true, mu_log, sigma_log, level), 4)
            m[f"width@{level:.2f}"] = round(targets.mean_interval_width(mu_log, sigma_log, level), 1)
        return m

    def evaluate(self, train_df, test_df, logits, artifact) -> dict[str, dict]:
        counts = test_df["eval_count"].to_numpy(dtype=float)
        mu_log, sigma_log = targets.model_to_log_params(logits[:, 0], logits[:, 1], artifact)

        # marginal baseline: train mean/std (no prompt signal)
        y_tr = targets.to_log(train_df["eval_count"])
        marg = (np.full(len(test_df), float(y_tr.mean())),
                np.full(len(test_df), float(y_tr.std()) or 1.0))

        # input-length baseline: linear fit of log-output on log-input length
        x_tr = targets.to_log(train_df["prompt_eval_count"])
        slope, intercept = np.polyfit(x_tr, y_tr, deg=1)
        resid_std = float((y_tr - (slope * x_tr + intercept)).std()) or 1.0
        inp = (slope * targets.to_log(test_df["prompt_eval_count"]) + intercept,
               np.full(len(test_df), resid_std))

        return {
            "distilbert":   self._metrics(counts, mu_log, sigma_log),
            "marginal":     self._metrics(counts, *marg),
            "input_length": self._metrics(counts, *inp),
        }


_TASKS: dict[str, type[Task]] = {
    ClassificationTask.name: ClassificationTask,
    RegressionTask.name: RegressionTask,
}

MODES = tuple(_TASKS)


def get_task(mode: str, n_buckets: int | None = None) -> Task:
    if mode not in _TASKS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODES)}")
    if mode == ClassificationTask.name:
        return ClassificationTask(n_buckets=n_buckets)
    return _TASKS[mode]()
