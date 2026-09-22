# Predicting LLM Token Cost from a Query

Given a text query, predict how many **output tokens** an LLM will generate to answer it — *before* running the model. Output tokens dominate inference cost, so predicting them from the prompt alone lets a developer estimate the cost of a query in advance.

## Approach

- **Prompt-only**: predict from the query text alone, with no access to the model's internal state and without generating any tokens first.
- **Output tokens only**: input tokens are trivial to count; the response length is the hard, useful part.
- **A range, not an exact count**: sampling randomness makes exact counts unpredictable, while a range is both achievable and sufficient for cost estimation.

A small pre-trained encoder (DistilBERT, ~66M params) is fine-tuned to do this. Two interchangeable **predictor modes** express "a range" differently — pick one with `--mode`:

| Mode | Head | What it predicts | Trained with |
|------|------|------------------|--------------|
| `classification` | 4 logits | One of four length **buckets** (quartile-based) | cross-entropy |
| `regression` | 2 logits | A per-prompt **mean ± std** of `log1p(tokens)` (heteroscedastic Gaussian) | Gaussian NLL |

`regression` is the default (`DEFAULT_MODE` in [`src/token_cost/config.py`](src/token_cost/config.py)). It turns each prompt into a median token estimate plus an asymmetric, never-negative interval (e.g. *≈340 tokens, 80% interval 120–910*), which composes cleanly for budgeting. `classification` is the simpler bucketed view.

## Datasets

Every dataset is treated identically: it declares a HuggingFace source and a
**labeler** (how `(prompt, output-token-count)` labels are produced), and all of
its files live under one uniform path. Labels come from the first user turn of
single-turn English conversations.

| Dataset | Source | Labeler | Output tokens |
|---------|--------|---------|---------------|
| `llama8b_generated` | [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | `generate` | Prompts answered locally by **Llama 3.1 8B** via Ollama; taken from the model's own count. |
| `wildchat` | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | `existing` | Counted from the existing ChatGPT responses with the GPT tokenizer (`tiktoken` `cl100k_base`). |

Adding a dataset is one entry in the `DATASETS` registry in [`src/token_cost/config.py`](src/token_cost/config.py); no other code changes are needed.

## Experiments

**Per-dataset** — train and evaluate a predictor on each dataset against its baselines. Each predictor is trained and used for a single target LLM, since output length depends on the answering model (a query's answer length differs from one model to the next).

The predictor mode is independent of the dataset: the same labeled data and the **same train/val/test split** feed either mode, so the two are directly comparable. Adding a mode is one entry in the `Task` registry in [`tasks.py`](src/token_cost/tasks.py).

### Baselines

- `classification`: **Majority bucket** (always the most common bucket) and **Input-length rule** (bucket from prompt length).
- `regression`: **Marginal** (predict the training mean/std — the no-signal floor) and **Input-length** (linear fit of log-output on log-input length; residual std as uncertainty).

### Metrics

- `classification`: **bucket accuracy**, **macro F1**, **off-by-one rate** (within one bucket of the truth).
- `regression`: **NLL** (Gaussian negative log-likelihood, lower is better), point-estimate **MAE/RMSE** (log and token space), and **calibration** — empirical coverage of the predicted 80%/95% intervals (should match the nominal level) plus mean interval width.

Success means clearly beating the baselines: higher accuracy/F1 for classification, lower NLL with well-calibrated coverage for regression.

## Pipeline

Each dataset produces `labeled.jsonl` via its labeler, after which the modeling
steps are identical for every dataset and parameterized by `--mode`:

```
                 ┌ generate labeler:  sample → generate ┐
download → labels ┤                                      ├→ make_dataset → train → evaluate → experiments
                 └ existing labeler:  build_labels       ┘      (--mode)    (--mode)  (--mode)
```

| Step | Module | Description |
|------|--------|-------------|
| `download` | `download_data.py` | Download each dataset's HF source into `data/<dataset>/raw/` (one-time). |
| `sample` | `sample_prompts.py` | *(generate labeler)* Sample single-turn English prompts. |
| `generate` | `generate_labels.py` | *(generate labeler)* Run prompts through the dataset's model via Ollama; record output tokens. |
| `build_labels` | `build_labels.py` | *(existing labeler)* Count output tokens from the dataset's existing responses. |
| `make_dataset` | `make_dataset.py` | Create the shared train/val/test split and the mode's artifact (bucket edges or target stats). |
| `train` | `train.py` | Fine-tune DistilBERT for the chosen mode. |
| `evaluate` | `evaluate.py` | Score the predictor and baselines on the test set. |
| `experiments` | `experiments.py` | Run every dataset for a mode and write a summary table. |
| `tasks` | `tasks.py` | The per-mode interface: head size, target, loss, metrics, baselines. |

The data layer (raw, prompts, labels) and the splits are **mode-independent and shared**; only the trained model and metrics are per-mode:

```
src/token_cost/                      pipeline package (config.py lives here)
scripts/                             report builders
docs/CLOUD.md                        AWS execution plan
reports/                             Word deliverables
data/<dataset>/raw/                  downloaded HF source
data/<dataset>/prompts.parquet       sampled prompts (generate labeler only)
data/<dataset>/labeled.jsonl         (prompt, output-token-count) labels
data/<dataset>/splits/               shared train/val/test parquet
models/<dataset>/<mode>/             fine-tuned encoder (one per mode)
metrics/                             evaluation JSON
report/                              generated figures
```

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com/) (for the Llama dataset).

```bash
uv sync

# One-time manual steps:
#  1. Accept dataset terms on HuggingFace (LMSYS-Chat-1M, WildChat-1M) and set HF_TOKEN in .env
#  2. ollama serve  &&  ollama pull llama3.1:8b
```

## Usage

```bash
# generate labeler — needs Ollama running
uv run python -m token_cost.download_data   --dataset llama8b_generated
uv run python -m token_cost.sample_prompts  --dataset llama8b_generated
uv run python -m token_cost.generate_labels --dataset llama8b_generated

# existing labeler — no model run needed
uv run python -m token_cost.download_data --dataset wildchat
uv run python -m token_cost.build_labels  --dataset wildchat

# Train and evaluate every dataset, choosing a predictor mode
uv run python -m token_cost.experiments --mode regression
uv run python -m token_cost.experiments --mode classification

# Or run a single step for one dataset + mode
uv run python -m token_cost.make_dataset --dataset wildchat --mode regression
uv run python -m token_cost.train        --dataset wildchat --mode regression
uv run python -m token_cost.evaluate     --dataset wildchat --mode regression
```

Steps share the split across modes and skip work that already exists, so switching modes only trains what's missing.

## Report

The final write-up covers the bucket classifier and the queue application, and lives in [`reports/Predicting the Length of an LLM Answer from the Prompt.docx`](reports). Every number in it comes from the metrics files, and its figures are rebuilt with:

```bash
uv run --with matplotlib python scripts/make_figures.py    # -> report/fig*.png
```

Key settings live in [`src/token_cost/config.py`](src/token_cost/config.py).

## Cloud

The next training run (ModernBERT-large, 2048-token prompts, WildChat-4.8M gpt-4o) is documented in [`docs/CLOUD.md`](docs/CLOUD.md). Clone, `uv sync`, then follow that file.
