"""Run Jev over the WildChat length buckets and compare with ModernBERT."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from token_cost import config, tasks
from token_cost.evaluate import print_table

MODEL = "jev-1.13.0"
N_BUCKETS = 8
MAX_PROMPT_TOKENS = 2_048
DEFAULT_CONCURRENCY = 16
PRICE_PER_MILLION_INPUT_TOKENS = 0.042
QUESTION_ID = "length_bucket"


def build_bucket_criteria(edges: list[int]) -> dict[str, str]:
    criteria: dict[str, str] = {}
    lower = 0
    for bucket, upper in enumerate(edges):
        criteria[str(bucket)] = (
            f"The answer would contain {lower} to {upper} output tokens."
        )
        lower = upper + 1
    criteria[str(len(edges))] = (
        f"The answer would contain {lower} or more output tokens."
    )
    return criteria


def load_cached_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    cached: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return cached
    with path.open() as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                cached[(record["split"], int(record["row_index"]))] = record
    return cached


def truncate_query(query: str, encoding: Any, max_tokens: int) -> str:
    tokens = encoding.encode(query)
    return query if len(tokens) <= max_tokens else encoding.decode(tokens[:max_tokens])


def split_names_for_scope(scope: str) -> tuple[str, ...]:
    return ("test",) if scope == "test" else ("test", "train", "val")


def evaluate_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    artifact: dict,
) -> dict[str, dict]:
    task = tasks.get_task("classification", n_buckets=artifact["n_buckets"])
    logits = np.zeros((len(predictions), artifact["n_buckets"]), dtype=float)
    logits[np.arange(len(predictions)), predictions] = 1.0
    return task.evaluate(train_df, test_df, logits, artifact)


async def classify_rows(
    client: Any,
    question: Any,
    rows: list[dict[str, Any]],
    out_path: Path,
    encoding: Any,
    max_prompt_tokens: int,
    concurrency: int,
) -> tuple[int, int, int]:
    semaphore = asyncio.Semaphore(concurrency)

    async def classify(row: dict[str, Any]) -> dict[str, Any] | None:
        async with semaphore:
            try:
                response = await client.system_one(
                    state=truncate_query(row["query"], encoding, max_prompt_tokens),
                    questions={QUESTION_ID: question},
                )
                answer = response.choices[QUESTION_ID]
                usage = response.usage
                return {
                    "split": row["split"],
                    "row_index": row["row_index"],
                    "id": str(row["id"]),
                    "pred_bucket": int(answer.choice),
                    "probabilities": answer.probabilities,
                    "confidence": answer.confidence,
                    "input_tokens": usage.input_tokens or 0,
                    "output_tokens": usage.output_tokens or 0,
                    "model": response.model,
                }
            except Exception as exc:
                print(f"\n[WARN] {row['split']}:{row['row_index']}: {exc}")
                return None

    completed = errors = input_tokens = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as output:
        for start in range(0, len(rows), 1_000):
            batch = [classify(row) for row in rows[start : start + 1_000]]
            for future in asyncio.as_completed(batch):
                record = await future
                if record is None:
                    errors += 1
                    continue
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                completed += 1
                input_tokens += int(record["input_tokens"])
                if completed % 100 == 0:
                    print(f"\r  completed {completed:,}/{len(rows):,}", end="", flush=True)
    if completed:
        print()
    return completed, errors, input_tokens


def main(
    scope: str = "test",
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    dry_run: bool = False,
) -> None:
    from tiktoken import get_encoding
    from typesafe_sdk import AsyncTypeSafeClient, Choice

    dataset = "wildchat"
    paths = config.dataset_paths(dataset)
    splits_dir = paths["splits_dir"]
    artifact = tasks.load_artifact(splits_dir / f"bucket_edges_k{N_BUCKETS}.json")
    split_names = split_names_for_scope(scope)
    frames = {
        split: pd.read_parquet(splits_dir / f"{split}.parquet")
        for split in split_names
    }
    cache_path = paths["labeled_file"].parent / "jev-1.13.0_k8_predictions.jsonl"
    cached = load_cached_predictions(cache_path)
    pending = [
        {
            "split": split,
            "row_index": row_index,
            "id": row["id"],
            "query": row["query"],
        }
        for split, frame in frames.items()
        for row_index, row in frame.iterrows()
        if (split, row_index) not in cached
    ]
    if limit is not None:
        pending = pending[:limit]

    print(
        f"[{dataset}] Jev {N_BUCKETS}-bucket scope={scope}: "
        f"{len(cached):,} cached, {len(pending):,} selected"
    )
    if dry_run:
        return

    errors = 0
    if pending:
        criteria = build_bucket_criteria(artifact["edges"])
        question = Choice(
            instructions=(
                "Predict the likely output-token length of a capable general-purpose "
                "assistant's answer to this user prompt. Choose the single most likely bucket."
            ),
            criteria=criteria,
        )
        encoding = get_encoding("cl100k_base")
        started = time.monotonic()

        async def run() -> tuple[int, int, int]:
            async with AsyncTypeSafeClient(model=MODEL) as client:
                return await classify_rows(
                    client,
                    question,
                    pending,
                    cache_path,
                    encoding,
                    MAX_PROMPT_TOKENS,
                    concurrency,
                )

        completed, errors, input_tokens = asyncio.run(run())
        elapsed = time.monotonic() - started
        cost = input_tokens * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000
        print(
            f"Completed {completed:,} calls with {errors} errors in {elapsed:.1f}s; "
            f"{input_tokens:,} input tokens; estimated cost ${cost:.4f}"
        )

    cached = load_cached_predictions(cache_path)
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    test_records = [cached.get(("test", index)) for index in range(len(test_df))]
    if any(record is None for record in test_records):
        print("Test split is incomplete; final metrics were not written.")
        if errors:
            raise SystemExit(1)
        return

    predictions = np.array(
        [record["pred_bucket"] for record in test_records if record is not None],
        dtype=int,
    )
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    results = evaluate_predictions(train_df, test_df, predictions, artifact)
    print_table(results)

    predictions_path = (
        paths["labeled_file"].parent / "predictions_classification_jev_k8.parquet"
    )
    pd.DataFrame(
        {"id": test_df["id"].to_numpy(), "pred_bucket": predictions}
    ).to_parquet(predictions_path, index=False)

    baseline_path = (
        config.METRICS_DIR
        / "wildchat_classification_ModernBERT-large_seq2048_k8.json"
    )
    baseline = json.loads(baseline_path.read_text())["results"]["model"]
    metrics_path = config.METRICS_DIR / "wildchat_classification_jev_k8.json"
    metrics_path.write_text(
        json.dumps(
            {
                "label": dataset,
                "mode": "classification",
                "model": MODEL,
                "n_buckets": N_BUCKETS,
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
                "results": results,
                "modernbert_8_bucket": baseline,
            },
            indent=2,
        )
    )
    print(f"Saved -> {predictions_path}")
    print(f"Saved -> {metrics_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="test", choices=("test", "all"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(
        scope=args.scope,
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )
