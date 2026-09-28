import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from token_cost.jev_classify import (
    build_bucket_criteria,
    classify_rows,
    evaluate_predictions,
    load_cached_predictions,
    split_names_for_scope,
    truncate_query,
)


class JevClassifyTest(unittest.TestCase):
    def test_build_bucket_criteria_uses_existing_edges(self):
        self.assertEqual(
            build_bucket_criteria([76, 229, 345]),
            {
                "0": "The answer would contain 0 to 76 output tokens.",
                "1": "The answer would contain 77 to 229 output tokens.",
                "2": "The answer would contain 230 to 345 output tokens.",
                "3": "The answer would contain 346 or more output tokens.",
            },
        )

    def test_load_cached_predictions_uses_split_and_row_index(self):
        records = [
            {"split": "test", "row_index": 0, "pred_bucket": 1},
            {"split": "train", "row_index": 0, "pred_bucket": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            cached = load_cached_predictions(path)

        self.assertEqual(cached[("test", 0)]["pred_bucket"], 1)
        self.assertEqual(cached[("train", 0)]["pred_bucket"], 2)

    def test_truncate_query_keeps_only_the_requested_token_prefix(self):
        class CharacterEncoding:
            @staticmethod
            def encode(text):
                return list(text)

            @staticmethod
            def decode(tokens):
                return "".join(tokens)

        self.assertEqual(truncate_query("abcdef", CharacterEncoding(), 4), "abcd")
        self.assertEqual(truncate_query("abc", CharacterEncoding(), 4), "abc")

    def test_all_scope_processes_test_split_first(self):
        self.assertEqual(split_names_for_scope("test"), ("test",))
        self.assertEqual(split_names_for_scope("all"), ("test", "train", "val"))

    def test_evaluate_predictions_reuses_classification_metrics(self):
        train_df = pd.DataFrame(
            {"eval_count": [10, 100, 250, 400], "prompt_eval_count": [5, 50, 200, 350]}
        )
        test_df = pd.DataFrame(
            {"eval_count": [20, 150, 300, 500], "prompt_eval_count": [10, 100, 260, 450]}
        )
        predictions = np.array([0, 1, 2, 3])

        results = evaluate_predictions(
            train_df,
            test_df,
            predictions,
            artifact={"edges": [76, 229, 345], "n_buckets": 4},
        )

        self.assertEqual(results["model"]["accuracy"], 1.0)
        self.assertEqual(results["model"]["macro_f1"], 1.0)
        self.assertEqual(results["model"]["off_by_one"], 1.0)


class JevClassifyAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_classify_rows_accepts_missing_usage_counts(self):
        class Client:
            async def system_one(self, state, questions):
                answer = SimpleNamespace(
                    choice="2",
                    probabilities={"0": 0.1, "1": 0.2, "2": 0.7},
                    confidence=0.7,
                )
                return SimpleNamespace(
                    choices={"length_bucket": answer},
                    usage=SimpleNamespace(input_tokens=None, output_tokens=None),
                    model="jev-1.13.0",
                )

        class CharacterEncoding:
            encode = staticmethod(list)
            decode = staticmethod(lambda tokens: "".join(tokens))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            completed, errors, input_tokens = await classify_rows(
                Client(),
                question=object(),
                rows=[{"split": "test", "row_index": 0, "id": "a", "query": "prompt"}],
                out_path=path,
                encoding=CharacterEncoding(),
                max_prompt_tokens=2_048,
                concurrency=1,
            )

        self.assertEqual((completed, errors, input_tokens), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
