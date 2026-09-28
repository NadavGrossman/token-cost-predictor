# Predicting output-token cost from a prompt

Predict how many tokens an LLM will generate for a query before you run it. The predictor sees only the prompt. Output length is what dominates inference cost.

Two heads, chosen with `--mode`:

| Mode | Prediction |
|------|------------|
| `classification` | One length bucket (4 or 8 quantile buckets) |
| `regression` (default) | A median token count, plus an 80% and 95% interval |

Each checkpoint is trained for one target LLM. `llama8b_generated` predicts Llama 3.1 8B (prompts from LMSYS-Chat-1M). `wildchat` predicts ChatGPT on English [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) conversations, with counts from the GPT tokenizer (`cl100k_base`). DistilBERT (256-token context) is the default encoder. ModernBERT-large (2048-token context) is used for the longer WildChat classification runs.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Download the checkpoints

The weight files are published on Hugging Face under [Nadi-Boss](https://huggingface.co/Nadi-Boss). They are public, so no token is required. This repo keeps the tokenizer and config files; `uv sync` is enough to download the weights into the folders the code reads.

```bash
uv run python <<'PY'
from huggingface_hub import snapshot_download

CHECKPOINTS = {
    "Nadi-Boss/token-cost-llama8b-distilbert-4bucket": "models/llama8b_generated/classification",
    "Nadi-Boss/token-cost-llama8b-distilbert-regression": "models/llama8b_generated/regression",
    "Nadi-Boss/token-cost-wildchat-distilbert-4bucket": "models/wildchat/classification",
    "Nadi-Boss/token-cost-wildchat-distilbert-regression": "models/wildchat/regression",
    "Nadi-Boss/token-cost-wildchat-modernbert-4bucket": "models/wildchat/classification/ModernBERT-large_seq2048",
    "Nadi-Boss/token-cost-wildchat-modernbert-8bucket": "models/wildchat/classification/ModernBERT-large_seq2048_k8",
}

for repo_id, local_dir in CHECKPOINTS.items():
    snapshot_download(repo_id, local_dir=local_dir)
    print(f"{repo_id} -> {local_dir}")
PY
```

To fetch a single checkpoint with the Hugging Face CLI:

```bash
hf download Nadi-Boss/token-cost-wildchat-modernbert-8bucket \
  --local-dir models/wildchat/classification/ModernBERT-large_seq2048_k8
```

| Hub repo | Saved to | Checkpoint |
|----------|----------|------------|
| [token-cost-llama8b-distilbert-4bucket](https://huggingface.co/Nadi-Boss/token-cost-llama8b-distilbert-4bucket) | `models/llama8b_generated/classification/` | DistilBERT, 4 buckets |
| [token-cost-llama8b-distilbert-regression](https://huggingface.co/Nadi-Boss/token-cost-llama8b-distilbert-regression) | `models/llama8b_generated/regression/` | DistilBERT, mean ± std |
| [token-cost-wildchat-distilbert-4bucket](https://huggingface.co/Nadi-Boss/token-cost-wildchat-distilbert-4bucket) | `models/wildchat/classification/` | DistilBERT, 4 buckets |
| [token-cost-wildchat-distilbert-regression](https://huggingface.co/Nadi-Boss/token-cost-wildchat-distilbert-regression) | `models/wildchat/regression/` | DistilBERT, mean ± std |
| [token-cost-wildchat-modernbert-4bucket](https://huggingface.co/Nadi-Boss/token-cost-wildchat-modernbert-4bucket) | `models/wildchat/classification/ModernBERT-large_seq2048/` | ModernBERT-large, 4 buckets |
| [token-cost-wildchat-modernbert-8bucket](https://huggingface.co/Nadi-Boss/token-cost-wildchat-modernbert-8bucket) | `models/wildchat/classification/ModernBERT-large_seq2048_k8/` | ModernBERT-large, 8 buckets |

Classification repos include `bucket_edges.json`. Regression repos include `target_stats.json`, which converts the two logits back into a token count. Training, evaluation, and the queue simulation also need `data/<dataset>/splits/`. Those labeled rows are large and are not in this repository.

## Project layout

```
src/token_cost/
  config.py           paths, encoders, datasets
  tasks.py            classification and regression
  targets.py          log-token math for the regression head
  make_dataset.py     70/15/15 split, bucket edges, target stats
  train.py            fine-tune; keeps an existing checkpoint unless --force
  evaluate.py         test metrics against simple baselines
  predict.py          one prediction per test prompt
  simulate_queue.py   queue simulation from those predictions
  experiments.py      make_dataset, then train, then evaluate
  jev_classify.py     Jev API baseline on the 8-bucket WildChat task
models/               checkpoints (weights downloaded from Hugging Face)
metrics/              evaluation JSON; the fine-tuned model is the "model" row
report/               figures
reports/              write-up
data/                 labeled rows and splits (not in git)
tests/                tests for the Jev comparison
```

Settings live in `src/token_cost/config.py`.

## Run

```bash
uv run python -m token_cost.experiments --mode regression
uv run python -m token_cost.experiments --mode classification

# ModernBERT, 8 buckets
uv run python -m token_cost.train    --dataset wildchat --mode classification --encoder modernbert --n-buckets 8
uv run python -m token_cost.evaluate --dataset wildchat --mode classification --encoder modernbert --n-buckets 8

uv run python -m token_cost.predict        --dataset wildchat --mode classification
uv run python -m token_cost.simulate_queue --dataset wildchat
```

`experiments` skips a step whose output is already on disk. Classification reports bucket accuracy, macro F1, and off-by-one rate. Regression reports Gaussian NLL, token error, and coverage of the 80% and 95% intervals.
