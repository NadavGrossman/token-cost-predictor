"""
Step 5 — Fine-tune an encoder for the chosen predictor mode.

The training loop is mode-agnostic: the per-row target, output-head size, loss
and validation score all come from the selected ``Task`` (see tasks.py). DistilBERT
@ 256 keeps ``models/<dataset>/<mode>/``; other encoders write a sibling folder.
Existing checkpoints are not overwritten unless ``--force`` is passed.

Usage:
    uv run python -m token_cost.train --dataset wildchat --mode classification
    uv run python -m token_cost.train --dataset wildchat48m --mode classification --encoder modernbert
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from token_cost import config
from token_cost import tasks
from token_cost.tasks import Task


class PromptDataset(Dataset):
    def __init__(self, df: pd.DataFrame, task: Task, artifact: dict):
        self.queries = df["query"].tolist()
        self.targets = task.make_targets(df, artifact)
        self.dtype = task.target_dtype

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        return {
            "query": self.queries[idx],
            "target": torch.tensor(self.targets[idx], dtype=self.dtype),
        }


def make_collate(tokenizer, max_len: int):
    """Pad to the longest prompt in the batch (capped at max_len), not every row to max_len."""
    def collate(rows: list[dict]) -> dict:
        enc = tokenizer(
            [r["query"] for r in rows],
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target": torch.stack([r["target"] for r in rows]),
        }
    return collate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader, returning (logits, targets) as numpy arrays."""
    model.eval()
    logits, tgts = [], []
    n = len(loader)
    log_every = max(1, n // 10) if n >= 20 else None
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask).logits
            logits.append(out.cpu().numpy())
            tgts.append(batch["target"].numpy())
            if log_every and i % log_every == 0:
                print(f"  inference {i}/{n}", flush=True)
    return np.concatenate(logits), np.concatenate(tgts)


def main(dataset: str, mode: str = config.DEFAULT_MODE, encoder: str | None = None,
         force: bool = False, max_seq_len: int | None = None,
         batch_size: int | None = None, n_buckets: int | None = None) -> None:
    spec = config.encoder_spec(encoder)
    base_model = spec["hf_id"]
    max_seq_len = spec["max_seq_len"] if max_seq_len is None else max_seq_len
    lr = spec["learning_rate"]
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size

    paths      = config.dataset_paths(dataset)
    splits_dir: Path = paths["splits_dir"]
    model_dir: Path  = config.model_dir(dataset, mode, encoder, max_seq_len, n_buckets)
    model_dir.mkdir(parents=True, exist_ok=True)

    task     = tasks.get_task(mode, n_buckets=n_buckets)
    artifact = tasks.load_artifact(splits_dir / task.artifact_filename)

    device = get_device()
    use_bf16 = device.type == "cuda"
    print(f"[{dataset}] mode={mode} encoder={encoder or config.DEFAULT_ENCODER} "
          f"model={base_model} seq={max_seq_len} batch={batch_size} "
          f"k={task.num_outputs if mode == 'classification' else '-'} "
          f"device={device} bf16={use_bf16}")
    print(f"  checkpoints -> {model_dir}")
    if (model_dir / "config.json").exists() and not force:
        raise SystemExit(
            f"[{dataset}] Refusing to overwrite existing checkpoint in {model_dir}. "
            "Pass --force to replace it, or use --encoder to write a new folder."
        )

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df   = pd.read_parquet(splits_dir / "val.parquet")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=task.num_outputs
    ).to(device)

    collate = make_collate(tokenizer, max_seq_len)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        PromptDataset(train_df, task, artifact),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        PromptDataset(val_df, task, artifact),
        batch_size=batch_size,
        collate_fn=collate,
        pin_memory=pin,
    )

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_score = float("inf")
    epochs_without_improvement = 0
    log_every = max(1, len(train_loader) // 5)

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tgt  = batch["target"].to(device)

            optimizer.zero_grad()
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    loss = task.compute_loss(logits, tgt)
            else:
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = task.compute_loss(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={total_loss/step:.4f}")

        val_logits, val_targets = collect_logits(model, val_loader, device)
        val_score = task.val_score(val_logits, val_targets)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_score={val_score:.4f}  (lower is better)")

        if val_score < best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  -> saved best model (val_score={best_val_score:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement ({epochs_without_improvement}/{config.PATIENCE})")
            if epochs_without_improvement >= config.PATIENCE:
                print(f"  early stop: patience {config.PATIENCE} reached.")
                break

    print(f"\n[{dataset}] ({mode}) training complete. Best val score: {best_val_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    parser.add_argument("--mode", default=config.DEFAULT_MODE, choices=list(tasks.MODES))
    parser.add_argument("--encoder", default=config.DEFAULT_ENCODER, choices=list(config.ENCODERS))
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-buckets", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing checkpoint in the target folder.")
    args = parser.parse_args()
    main(dataset=args.dataset, mode=args.mode, encoder=args.encoder,
         force=args.force, max_seq_len=args.max_seq_len, batch_size=args.batch_size,
         n_buckets=args.n_buckets)
