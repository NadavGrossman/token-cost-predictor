"""
Experiments orchestrator — train + evaluate a predictor on each dataset for the
chosen mode, then print a summary table.

Usage:
    uv run python -m token_cost.experiments                       # default mode
    uv run python -m token_cost.experiments --mode classification
    uv run python -m token_cost.experiments --mode regression
"""
import argparse
import json

from token_cost import config
from token_cost import tasks
from token_cost.make_dataset import main as make_dataset
from token_cost.train import main as train
from token_cost.evaluate import main as evaluate

def dataset_ready(name: str) -> bool:
    return config.dataset_paths(name)["labeled_file"].exists()

def artifacts_ready(name: str, task: tasks.Task) -> bool:
    splits_dir = config.dataset_paths(name)["splits_dir"]
    splits = all((splits_dir / f"{s}.parquet").exists() for s in ("train", "val", "test"))
    return splits and (splits_dir / task.artifact_filename).exists()

def model_ready(name: str, mode: str) -> bool:
    return (config.model_dir(name, mode) / "config.json").exists()


def run_per_dataset(name: str, mode: str) -> dict:
    print(f"\n{'#'*60}\n# Dataset: {name}  |  mode: {mode}\n{'#'*60}")
    task = tasks.get_task(mode)

    if artifacts_ready(name, task):
        print(f"[{name}] Splits + {mode} artifact exist — skipping make_dataset.")
    else:
        make_dataset(dataset=name, mode=mode, max_rows=config.DATASETS[name]["max_rows"])

    if model_ready(name, mode):
        print(f"[{name}] Trained {mode} model exists — skipping train.")
    else:
        train(dataset=name, mode=mode)

    metrics_path = config.metrics_file(name, mode)
    if metrics_path.exists():
        print(f"[{name}] Metrics exist at {metrics_path} — skipping evaluate.")
        return json.loads(metrics_path.read_text()).get("results", {})
    return evaluate(dataset=name, mode=mode)


def main(mode: str = config.DEFAULT_MODE) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    task = tasks.get_task(mode)

    all_results: dict[str, dict] = {}
    for name, cfg in config.DATASETS.items():
        if cfg.get("skip_default_experiments"):
            print(f"\n[{name}] skipped by default (cloud dataset).")
            continue
        if not dataset_ready(name):
            print(f"\n[{name}] labeled.jsonl not found — skipping. "
                  f"(Run generate_labels or build_labels first.)")
            continue
        all_results[name] = run_per_dataset(name, mode)

    summary_file = config.METRICS_DIR / f"summary_{mode}.json"
    if summary_file.exists():
        print(f"\nSummary exists at {summary_file} — not overwriting.")
    else:
        with summary_file.open("w") as f:
            json.dump(all_results, f, indent=2)

    cols = task.summary_cols
    header = f"{'experiment':<35}" + "".join(f"{c:>12}" for c in cols)
    print(f"\n{'='*len(header)}")
    print(f"SUMMARY — DistilBERT ({mode}) across all experiments")
    print(f"{'='*len(header)}\n{header}\n{'-'*len(header)}")
    for exp_name, res in all_results.items():
        db = res.get("distilbert")
        if db:
            print(f"{exp_name:<35}" + "".join(f"{db[c]:>12}" for c in cols))
    print(f"{'='*len(header)}")
    print(f"\nFull results -> {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=config.DEFAULT_MODE, choices=list(tasks.MODES))
    args = parser.parse_args()
    main(mode=args.mode)
