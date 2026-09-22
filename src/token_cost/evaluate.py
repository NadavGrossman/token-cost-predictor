"""
Step 6 — Evaluate the trained predictor and its baselines on the test set.

Mode-agnostic: runs the fine-tuned model over the test split to get raw logits,
then hands them to the selected ``Task`` which computes the appropriate metrics
(classification: accuracy / macro-F1 / off-by-one; regression: NLL / errors /
calibrated coverage) for the model and every baseline.

Usage:
    uv run python -m token_cost.evaluate --dataset wildchat --mode regression
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from token_cost import config
from token_cost import tasks
from token_cost.tasks import Task
from token_cost.train import PromptDataset, collect_logits, get_device, make_collate
from torch.utils.data import DataLoader


def model_logits(test_df: pd.DataFrame, model_dir: Path, task: Task, artifact: dict,
                 max_seq_len: int | None = None, batch_size: int | None = None) -> np.ndarray:
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    seq = max_seq_len or config.MAX_SEQ_LEN
    bs = config.BATCH_SIZE if batch_size is None else batch_size
    loader = DataLoader(
        PromptDataset(test_df, task, artifact),
        batch_size=bs,
        collate_fn=make_collate(tokenizer, seq),
        pin_memory=device.type == "cuda",
    )
    logits, _ = collect_logits(model, loader, device)
    return logits


def print_table(results: dict[str, dict]) -> None:
    cols = list(next(iter(results.values())).keys())
    header = f"{'method':<14}" + "".join(f"{c:>12}" for c in cols)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for name, m in results.items():
        print(f"{name:<14}" + "".join(f"{m[c]:>12}" for c in cols))
    print(sep)


def main(dataset: str, mode: str = config.DEFAULT_MODE, encoder: str | None = None,
         force: bool = False, max_seq_len: int | None = None,
         batch_size: int | None = None, n_buckets: int | None = None) -> dict:
    spec = config.encoder_spec(encoder)
    max_seq_len = spec["max_seq_len"] if max_seq_len is None else max_seq_len
    paths        = config.dataset_paths(dataset)
    splits_dir: Path   = paths["splits_dir"]
    model_path: Path   = config.model_dir(dataset, mode, encoder, max_seq_len, n_buckets)
    metrics_path: Path = config.metrics_file(dataset, mode, encoder, max_seq_len, n_buckets)
    if metrics_path.exists() and not force:
        raise SystemExit(
            f"[{dataset}] Refusing to overwrite existing metrics at {metrics_path}. "
            "Pass --force to replace them."
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    task     = tasks.get_task(mode, n_buckets=n_buckets)
    artifact = tasks.load_artifact(splits_dir / task.artifact_filename)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    test_df  = pd.read_parquet(splits_dir / "test.parquet")

    print(f"[{dataset}] ({mode}) encoder={encoder or config.DEFAULT_ENCODER} running model inference …")
    logits = model_logits(test_df, model_path, task, artifact,
                          max_seq_len=max_seq_len, batch_size=batch_size)
    all_results = task.evaluate(train_df, test_df, logits, artifact)

    print_table(all_results)

    with metrics_path.open("w") as f:
        json.dump({
            "label": dataset, "mode": mode,
            "encoder": encoder or config.DEFAULT_ENCODER,
            "n_buckets": n_buckets or (artifact.get("n_buckets") if mode == "classification" else None),
            "results": all_results,
        }, f, indent=2)
    print(f"Saved -> {metrics_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    parser.add_argument("--mode", default=config.DEFAULT_MODE, choices=list(tasks.MODES))
    parser.add_argument("--encoder", default=config.DEFAULT_ENCODER, choices=list(config.ENCODERS))
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-buckets", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing metrics JSON.")
    args = parser.parse_args()
    main(dataset=args.dataset, mode=args.mode, encoder=args.encoder,
         force=args.force, max_seq_len=args.max_seq_len, batch_size=args.batch_size,
         n_buckets=args.n_buckets)
