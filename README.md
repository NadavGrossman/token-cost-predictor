# Predicting LLM Token Cost from a Query

Given a text query, predict how many **output tokens** an LLM will generate to answer it — *before* running the model. Output tokens dominate inference cost, so predicting them from the prompt alone lets a developer estimate the cost of a query in advance.

This repository contains the predictor code, the fine-tuned checkpoints, and the results. The labeled datasets are large and are not included. Re-training or re-scoring a checkpoint needs `data/<dataset>/splits/` (and `labeled.jsonl` if the splits are not already built).

## Approach

- **Prompt-only**: predict from the query text alone, with no access to the model's internal state.
- **Output tokens only**: input tokens are trivial to count; the response length is the hard, useful part.
- **A range, not an exact count**: sampling randomness makes exact counts unpredictable, while a range is both achievable and sufficient for cost estimation.

Two encoders are fine-tuned. **DistilBERT** (~66M parameters, 256-token context) is the default. **ModernBERT-large** (2048-token context) is used for the longer WildChat classification runs. Two interchangeable **predictor modes** express "a range" differently — pick one with `--mode`:

| Mode | Head | What it predicts | Trained with |
|------|------|------------------|--------------|
| `classification` | 4 or 8 logits | One length **bucket** (quantile edges from the training set) | cross-entropy |
| `regression` | 2 logits | A per-prompt **mean ± std** of `log1p(tokens)` (heteroscedastic Gaussian) | Gaussian NLL |

`regression` is the default (`DEFAULT_MODE` in [`src/token_cost/config.py`](src/token_cost/config.py)). It turns each prompt into a median token estimate plus an asymmetric, never-negative interval (e.g. *≈340 tokens, 80% interval 120–910*), which composes cleanly for budgeting. `targets.predict_summary` converts a regression prediction into that token-space forecast. `classification` is the simpler bucketed view.

## Datasets

Each predictor is trained for a single target LLM, because answer length depends on the model that writes the answer. Labels are the output-token count of the first user turn in single-turn English conversations.

| Dataset | Target LLM | Where the counts come from |
|---------|------------|----------------------------|
| `llama8b_generated` | Llama 3.1 8B | Prompts from [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), answered by Llama 3.1 8B. The count is the model's own. |
| `wildchat` | ChatGPT | [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M). Counts use the GPT tokenizer (`cl100k_base`). Training uses a 300,000-row cap. |

The same labeled rows and the **same train/val/test split** feed both modes, so the two are directly comparable. Adding a mode is one entry in the `Task` registry in [`tasks.py`](src/token_cost/tasks.py).

## Experiments

### Baselines

- `classification`: **Majority bucket** (always the most common bucket) and **Input-length rule** (bucket from prompt length).
- `regression`: **Marginal** (predict the training mean/std — the no-signal floor) and **Input-length** (linear fit of log-output on log-input length; residual std as uncertainty).

### Metrics

In the metrics JSON, the fine-tuned model is the `model` row.

- `classification`: **bucket accuracy**, **macro F1**, **off-by-one rate** (within one bucket of the truth).
- `regression`: **NLL** (Gaussian negative log-likelihood, lower is better), point-estimate **MAE/RMSE** (log and token space), and **calibration** — empirical coverage of the predicted 80%/95% intervals (should match the nominal level) plus mean interval width.

Success means beating the baselines: higher accuracy/F1 for classification, lower NLL with well-calibrated coverage for regression.

## Checkpoints

| Path | Encoder | Task |
|------|---------|------|
| `models/llama8b_generated/classification/` | DistilBERT | 4 buckets |
| `models/llama8b_generated/regression/` | DistilBERT | mean ± std |
| `models/wildchat/classification/` | DistilBERT | 4 buckets |
| `models/wildchat/regression/` | DistilBERT | mean ± std |
| `models/wildchat/classification/ModernBERT-large_seq2048/` | ModernBERT-large | 4 buckets, 2048 tokens |
| `models/wildchat/classification/ModernBERT-large_seq2048_k8/` | ModernBERT-large | 8 buckets, 2048 tokens |

An existing checkpoint is left in place unless `--force` is passed.

## Results

| Path | Contents |
|------|----------|
| `metrics/` | Test metrics for each dataset, mode, and encoder, plus the queue-simulation runs |
| `report/` | Figures and the summary stats behind the write-up |
| `reports/` | The write-up (Word, PDF, and slides) |

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# Train and evaluate every dataset for one mode.
# Skips a step when its artifact already exists.
uv run python -m token_cost.experiments --mode regression
uv run python -m token_cost.experiments --mode classification

# Or run a single step for one dataset + mode
uv run python -m token_cost.make_dataset --dataset wildchat --mode regression
uv run python -m token_cost.train        --dataset wildchat --mode regression
uv run python -m token_cost.evaluate     --dataset wildchat --mode regression

# ModernBERT classification (4-way, or 8-way with --n-buckets 8)
uv run python -m token_cost.train    --dataset wildchat --mode classification --encoder modernbert
uv run python -m token_cost.evaluate --dataset wildchat --mode classification --encoder modernbert

# Cache per-prompt test predictions, then run the queue simulation
uv run python -m token_cost.predict        --dataset wildchat --mode classification
uv run python -m token_cost.simulate_queue --dataset wildchat
```

`make_dataset` builds a 70/15/15 split from `data/<dataset>/labeled.jsonl` and writes the mode's artifact (bucket edges or target stats) next to it. Later steps reuse that split.

## Layout

```
src/token_cost/            predictor package (config.py lives here)
models/<dataset>/<mode>/   fine-tuned encoder
metrics/                   evaluation and queue-simulation JSON
report/                    figures
reports/                   write-up
data/<dataset>/            labeled rows and splits (not included)
```

Key settings live in [`src/token_cost/config.py`](src/token_cost/config.py).
