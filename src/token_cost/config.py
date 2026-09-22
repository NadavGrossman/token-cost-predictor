"""Central config — all tuneable knobs in one place."""
from pathlib import Path

# Repo root is two levels above this file (src/token_cost/config.py).
ROOT = Path(__file__).resolve().parents[2]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
METRICS_DIR = ROOT / "metrics"

# ── Shared settings ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Predictor mode ────────────────────────────────────────────────────────────
# "classification" -> length buckets (cross-entropy);
# "regression"     -> heteroscedastic Gaussian (per-prompt mean + std).
# Every step takes --mode; this is just the default when it is omitted.
DEFAULT_MODE = "regression"

# ── Bucketing (classification mode) ───────────────────────────────────────────
N_BUCKETS = 4   # quartile-based; edges computed from training data

# ── Target / uncertainty (regression mode) ────────────────────────────────────
# Output-token count is modeled as log-normal: the predictor outputs the mean and
# log-std of log1p(count). These are the central coverage levels whose calibration
# (and interval width) we report at evaluation time.
COVERAGE_LEVELS = (0.80, 0.95)

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

# ── Encoder presets ───────────────────────────────────────────────────────────
# distilbert: default runs, 256-token context.
# modernbert: WildChat classification at 2048 tokens, with 4- and 8-way buckets.
ENCODERS: dict[str, dict] = {
    "distilbert": {
        "hf_id": "distilbert-base-uncased",
        "max_seq_len": 256,
        "learning_rate": 2e-5,
    },
    "modernbert": {
        "hf_id": "answerdotai/ModernBERT-large",
        "max_seq_len": 2048,
        "learning_rate": 3e-5,
    },
}
DEFAULT_ENCODER = "distilbert"
MAX_SEQ_LEN   = ENCODERS[DEFAULT_ENCODER]["max_seq_len"]
BATCH_SIZE    = 32
EPOCHS        = 6

# ── Regression-specific training controls ─────────────────────────────────────
# LOG_SIGMA_CLAMP: prevents sigma collapse (model pushing log σ to -∞ on training
# data to cheat NLL). Lower bound of -2 keeps min σ ≈ 0.14 in log space, which
# is still very tight but not degenerate. Upper bound of 4 allows very wide
# intervals without instability.
LOG_SIGMA_CLAMP = (-2.0, 4.0)

# PATIENCE: stop training if val score has not improved for this many consecutive
# epochs. Avoids the wasteful divergence seen after the best checkpoint.
PATIENCE = 2

# BETA_NLL (Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic Uncertainty
# Estimation"): weight each example's Gaussian NLL by a detached (sigma^2)^beta.
# Plain NLL (beta=0) down-weights high-error prompts, so the model leans into the
# easy ones and drives sigma overconfidently low -> the train↓ / val↑ divergence
# we observed. beta=0.5 cancels that re-weighting; 0 recovers standard NLL.
BETA_NLL = 0.5

# ── Queue-scheduling simulation ───────────────────────────────────────────────
# Service time stands in for a real LLM: decoding is ~constant time per output
# token, so t = overhead + output_tokens / rate.
DECODE_TOKENS_PER_SEC = 50.0
SERVICE_OVERHEAD_SEC  = 0.5

# ── Dataset registry ──────────────────────────────────────────────────────────
# max_rows caps the training pool. None keeps every labeled row.
# Provenance of each dataset is described in the README.
DATASETS: dict[str, dict] = {
    "llama8b_generated": {"max_rows": None},
    "wildchat":          {"max_rows": 300_000},
}


def dataset_paths(name: str) -> dict[str, Path]:
    """Shared file-system layout for a dataset.

    Labels and the train/val/test splits are identical across predictor modes.
    The trained model and metrics are mode-specific (see ``model_dir`` /
    ``metrics_file``). Each mode's small derived artifact (bucket edges or
    target stats) lives in ``splits_dir`` under its own name.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    base = DATA_DIR / name
    return {
        "labeled_file": base / "labeled.jsonl",
        "splits_dir":   base / "splits",
    }


def encoder_spec(encoder: str | None = None) -> dict:
    """Resolve an encoder preset. Unknown names raise ValueError with the allow-list."""
    name = encoder or DEFAULT_ENCODER
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODERS)}")
    return ENCODERS[name]


def run_slug(encoder: str | None = None, max_seq_len: int | None = None,
             n_buckets: int | None = None) -> str | None:
    """None for the DistilBERT@256 k=4 layout so those artifacts stay put."""
    name = encoder or DEFAULT_ENCODER
    spec = encoder_spec(name)
    seq = spec["max_seq_len"] if max_seq_len is None else max_seq_len
    k = N_BUCKETS if n_buckets is None else n_buckets
    if name == DEFAULT_ENCODER and seq == spec["max_seq_len"] and k == 4:
        return None
    hf_tail = spec["hf_id"].rsplit("/", 1)[-1]
    slug = f"{hf_tail}_seq{seq}"
    if k != 4:
        slug = f"{slug}_k{k}"
    return slug


def model_dir(name: str, mode: str, encoder: str | None = None,
              max_seq_len: int | None = None, n_buckets: int | None = None) -> Path:
    """Per-mode trained model directory (so modes never overwrite each other).

    DistilBERT@256 k=4 keeps ``models/<dataset>/<mode>/``. Other encoders write
    under ``models/<dataset>/<mode>/<slug>/``.
    """
    base = MODELS_DIR / name / mode
    slug = run_slug(encoder, max_seq_len, n_buckets)
    return base if slug is None else base / slug


def metrics_file(name: str, mode: str, encoder: str | None = None,
                 max_seq_len: int | None = None, n_buckets: int | None = None) -> Path:
    """Per-mode evaluation results file. Non-default encoders get a suffix."""
    slug = run_slug(encoder, max_seq_len, n_buckets)
    if slug is None:
        return METRICS_DIR / f"{name}_{mode}.json"
    return METRICS_DIR / f"{name}_{mode}_{slug}.json"


def predictions_file(name: str, mode: str, encoder: str | None = None,
                     max_seq_len: int | None = None, n_buckets: int | None = None) -> Path:
    """Per-mode, per-prompt test-set predictions (cached model outputs)."""
    slug = run_slug(encoder, max_seq_len, n_buckets)
    if slug is None:
        return DATA_DIR / name / f"predictions_{mode}.parquet"
    return DATA_DIR / name / f"predictions_{mode}_{slug}.parquet"
