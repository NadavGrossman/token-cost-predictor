"""
Step 6b — Cache per-prompt test-set predictions to a parquet file.

``evaluate`` only stores aggregate metrics; downstream consumers (e.g. the
queue-scheduling simulation) need the model's prediction for each prompt.
This runs the fine-tuned model once over the test split and saves one row per
prompt: the id plus the mode's prediction (classification: ``pred_bucket``;
regression: ``pred_median`` token count).

Usage:
    uv run python -m token_cost.predict --dataset wildchat --mode classification
"""
import argparse
from pathlib import Path

import pandas as pd

from token_cost import config
from token_cost import targets, tasks
from token_cost.evaluate import model_logits

def main(dataset: str, mode: str = config.DEFAULT_MODE, encoder: str | None = None,
         force: bool = False, max_seq_len: int | None = None,
         n_buckets: int | None = None, batch_size: int | None = None) -> Path:
    spec       = config.encoder_spec(encoder)
    max_seq_len = spec["max_seq_len"] if max_seq_len is None else max_seq_len
    paths      = config.dataset_paths(dataset)
    splits_dir = paths["splits_dir"]
    out_path   = config.predictions_file(dataset, mode, encoder, max_seq_len, n_buckets)
    if out_path.exists() and not force:
        raise SystemExit(
            f"[{dataset}] Refusing to overwrite existing predictions at {out_path}. "
            "Pass --force to replace them."
        )

    task     = tasks.get_task(mode, n_buckets=n_buckets)
    artifact = tasks.load_artifact(splits_dir / task.artifact_filename)
    test_df  = pd.read_parquet(splits_dir / "test.parquet")

    print(f"[{dataset}] ({mode}) encoder={encoder or config.DEFAULT_ENCODER} "
          f"running model inference over {len(test_df)} test prompts …")
    logits = model_logits(
        test_df, config.model_dir(dataset, mode, encoder, max_seq_len, n_buckets), task, artifact,
        max_seq_len=max_seq_len, batch_size=batch_size,
    )

    out = pd.DataFrame({"id": test_df["id"].to_numpy()})
    if mode == "classification":
        out["pred_bucket"] = logits.argmax(axis=1)
    else:
        mu_log, _ = targets.model_to_log_params(logits[:, 0], logits[:, 1], artifact)
        out["pred_median"] = targets.to_counts(mu_log)

    out.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    parser.add_argument("--mode", default=config.DEFAULT_MODE, choices=list(tasks.MODES))
    parser.add_argument("--encoder", default=config.DEFAULT_ENCODER, choices=list(config.ENCODERS))
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--n-buckets", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing predictions parquet.")
    args = parser.parse_args()
    main(dataset=args.dataset, mode=args.mode, encoder=args.encoder,
         force=args.force, max_seq_len=args.max_seq_len, n_buckets=args.n_buckets,
         batch_size=args.batch_size)
