"""Central config — all tuneable knobs in one place."""
import os
from pathlib import Path

# Repo root is two levels above this file (src/token_cost/config.py).
ROOT = Path(__file__).resolve().parents[2]

# Load .env if present (sets HF_TOKEN etc. without requiring python-dotenv)
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
METRICS_DIR = ROOT / "metrics"
S3_BUCKET   = os.environ.get("S3_BUCKET", "cs-workshop-token-cost-343798904084")

# ── Shared settings ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Predictor mode ────────────────────────────────────────────────────────────
# "classification" -> 4 length buckets (cross-entropy);
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

# ── Label generation (Ollama "generate" labeler) ──────────────────────────────
MAX_NEW_TOKENS: int | None = None   # cap on generated tokens; None = unbounded

# ── Existing-response labeler (token counting) ────────────────────────────────
REFERENCE_TOKENIZER = "cl100k_base"   # tiktoken / GPT tokenizer

# ── Encoder presets ───────────────────────────────────────────────────────────
# Default stays DistilBERT @ 256 so existing checkpoints/metrics keep working.
# Cloud ablation uses --encoder modernbert (isolated model/metrics paths).
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
BASE_MODEL    = ENCODERS[DEFAULT_ENCODER]["hf_id"]
MAX_SEQ_LEN   = ENCODERS[DEFAULT_ENCODER]["max_seq_len"]
BATCH_SIZE    = 32
EPOCHS        = 6
LEARNING_RATE = ENCODERS[DEFAULT_ENCODER]["learning_rate"]

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
# Adding a dataset is one entry here; all paths derive from the name (dataset_paths).
# labeler: "generate" = sample prompts + run a model (Ollama); "existing" = count
#          output tokens from responses already present in the dataset.
DATASETS: dict[str, dict] = {
    "llama8b_generated": {
        "hf_dataset":   "lmsys/lmsys-chat-1m",
        "labeler":      "generate",
        "ollama_model": "llama3.1:8b",
        "n_prompts":    20_000,
        "max_rows":     None,
    },
    "wildchat": {
        "hf_dataset": "allenai/WildChat-1M",
        "labeler":    "existing",
        "max_rows":   300_000,   # training cap; counting is uncapped
    },
    # Cloud dataset: new directory, never writes into data/wildchat/.
    "wildchat48m": {
        "hf_dataset": "allenai/WildChat-4.8M",
        "labeler":    "existing",
        "max_rows":   None,
        "stream_from_hf": True,
        "include_model_prefixes": ["gpt-4o"],
        "exclude_model_prefixes": ["o1-preview", "o1-mini", "gpt-4o-mini"],
        "dedupe_by": "query",
        "skip_default_experiments": True,
    },
}


def dataset_paths(name: str) -> dict[str, Path]:
    """Mode-agnostic, shared file-system layout for a dataset.

    Raw data, prompts, labels and the train/val/test splits are identical across
    predictor modes — only the trained model and metrics are mode-specific (see
    ``model_dir`` / ``metrics_file``), and each mode's small derived artifact
    (bucket edges or target stats) lives in ``splits_dir`` under its own name.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    base = DATA_DIR / name
    return {
        "raw_dir":      base / "raw",
        "prompts_file": base / "prompts.parquet",
        "labeled_file": base / "labeled.jsonl",
        "splits_dir":   base / "splits",
    }


def encoder_spec(encoder: str | None = None) -> dict:
    """Resolve an encoder preset. Unknown names raise KeyError with the allow-list."""
    name = encoder or DEFAULT_ENCODER
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODERS)}")
    return ENCODERS[name]


def run_slug(encoder: str | None = None, max_seq_len: int | None = None,
             n_buckets: int | None = None) -> str | None:
    """None for the legacy DistilBERT@256 k=4 layout so old artifacts stay put."""
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


def datasets_with_labeler(labeler: str) -> list[str]:
    """Names of datasets whose labels are produced by the given labeler."""
    return [name for name, cfg in DATASETS.items() if cfg["labeler"] == labeler]
