"""
Create the shared train/val/test splits and the per-mode artifact.

The 70/15/15 split is deterministic and identical for both predictor modes, so
the two are compared on the exact same rows. Only the small derived artifact
differs by mode:
  classification -> bucket_edges.json   (quartile cut points)
  regression     -> target_stats.json   (log-target mean/std)

Existing splits are reused; re-running for a second mode just adds that mode's
artifact.

Usage:
    uv run python -m token_cost.make_dataset --dataset wildchat --mode regression
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from token_cost import config
from token_cost import tasks


def _load_or_create_splits(dataset: str, splits_dir: Path, max_rows: int | None) -> pd.DataFrame:
    """Return the train split, (re)creating shared splits on disk if needed."""
    paths = {s: splits_dir / f"{s}.parquet" for s in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()):
        print(f"[{dataset}] Reusing existing splits in {splits_dir}")
        return pd.read_parquet(paths["train"])

    df = pd.read_json(config.dataset_paths(dataset)["labeled_file"], lines=True)
    print(f"[{dataset}] Loaded {len(df):,} labeled rows")
    print(f"  eval_count stats:\n{df['eval_count'].describe().to_string()}")

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=config.RANDOM_SEED).reset_index(drop=True)
        print(f"  capped to {len(df):,} rows (max_rows={max_rows:,})")

    train_val, test = train_test_split(
        df, test_size=1 - config.TRAIN_FRAC - config.VAL_FRAC, random_state=config.RANDOM_SEED
    )
    val_frac_of_remainder = config.VAL_FRAC / (config.TRAIN_FRAC + config.VAL_FRAC)
    train, val = train_test_split(
        train_val, test_size=val_frac_of_remainder, random_state=config.RANDOM_SEED
    )

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_parquet(paths[name], index=False)
        print(f"  {name}: {len(split_df):,} rows -> {paths[name]}")
    return train


def main(dataset: str, mode: str = config.DEFAULT_MODE, max_rows: int | None = None,
         n_buckets: int | None = None) -> None:
    if max_rows is None:
        max_rows = config.DATASETS[dataset]["max_rows"]
    splits_dir: Path = config.dataset_paths(dataset)["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    task  = tasks.get_task(mode, n_buckets=n_buckets)
    train = _load_or_create_splits(dataset, splits_dir, max_rows)

    artifact_file = splits_dir / task.artifact_filename
    if artifact_file.exists() and task.artifact_filename == "bucket_edges.json":
        print(f"[{dataset}] keeping existing 4-bucket artifact {artifact_file}")
        return

    artifact = task.prepare(train)
    tasks.save_artifact(artifact, artifact_file)
    print(f"[{dataset}] ({mode}) artifact: {artifact}")
    print(f"  saved -> {artifact_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    parser.add_argument("--mode", default=config.DEFAULT_MODE, choices=list(tasks.MODES))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-buckets", type=int, default=None)
    args = parser.parse_args()
    main(dataset=args.dataset, mode=args.mode, max_rows=args.max_rows,
         n_buckets=args.n_buckets)
