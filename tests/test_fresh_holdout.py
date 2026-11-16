"""Fresh holdout must replay published splits and avoid any recorded exposure."""
import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from waferbench.common import LABELS, read_jsonl, write_json, write_jsonl
from waferbench.data import prepare_mixed
from waferbench.fresh_holdout import prepare_fresh_mixed_holdout


class FreshHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.npz"
        generator = np.random.default_rng(100)
        # Near-full's test pool is exhausted by the original cap, and must be skipped.
        counts = [200, 200, 134]
        labels = [["center", "scratch"], ["loc", "scratch"], ["near_full"]]
        maps = generator.integers(0, 3, size=(sum(counts), 7, 7), dtype=np.uint8)
        targets = np.asarray([[int(name in truth) for name in LABELS]
                              for count, truth in zip(counts, labels) for _ in range(count)], dtype=np.uint8)
        np.savez_compressed(self.source, maps, targets)
        self.reference = self.root / "data" / "mixedwm38" / "manifest.jsonl"
        with patch("waferbench.data.ROOT", self.root), patch("waferbench.common.ROOT", self.root):
            prepare_mixed(self.source, self.reference.parent)
        self.reference_bytes = self.reference.read_bytes()
        self.root_patch = patch("waferbench.fresh_holdout.ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.out = self.root / "data" / "fresh"

    def prepare(self, out=None, **options):
        return prepare_fresh_mixed_holdout(self.source, self.reference, out or self.out,
                                          per_combo=options.pop("per_combo", 3), **options)

    def test_exact_replay_fresh_mixed_selection_and_source_alignment(self):
        report = self.prepare()
        fresh, original = read_jsonl(self.out / "manifest.jsonl"), read_jsonl(self.reference)
        self.assertEqual(report["n"], 6)
        self.assertEqual(report["mixed_combinations"], 2)
        self.assertEqual(Counter(row["combo"] for row in fresh), {"center+scratch": 3, "loc+scratch": 3})
        self.assertTrue(report["reference_replay"]["exact_match"])
        self.assertTrue(report["all_audited_overlaps_zero"])
        self.assertEqual(set(report["overlap_counts"].values()), {0})
        self.assertEqual(report["status"], "prepared_unscored")
        self.assertEqual(report["model_predictions_computed"], 0)
        self.assertEqual(report["api_calls_made"], 0)
        self.assertEqual(self.reference.read_bytes(), self.reference_bytes)
        self.assertFalse({row["id"] for row in fresh} & {row["id"] for row in original})
        self.assertFalse({row["group_id"] for row in fresh} & {row["group_id"] for row in original})
        with np.load(self.source, allow_pickle=False) as archive:
            source = archive["arr_0"]
        for row in fresh:
            self.assertEqual(row["split"], "test")
            self.assertGreaterEqual(len(row["labels"]), 2)
            np.testing.assert_array_equal(np.load(self.root / row["map_path"], allow_pickle=False), source[row["source_index"]])
        self.assertFalse((self.out / "images").exists())
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_manifest_identity_or_seed_drift_is_rejected_before_writing(self):
        rows = read_jsonl(self.reference)
        for field, value in [("source_index", 999), ("group_id", "tampered"), ("split", "val")]:
            changed = copy.deepcopy(rows)
            changed[0][field] = value if changed[0][field] != value else "test"
            write_jsonl(self.reference, changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.out.exists())
        self.reference.write_bytes(self.reference_bytes)
        with self.assertRaises(ValueError):
            self.prepare(seed=43)
        self.assertFalse(self.out.exists())

    def test_other_prepared_dataset_and_api_config_errors_raw_and_predictions_are_excluded(self):
        preview = self.root / "preview-outside-dataset-audit"
        self.prepare(out=preview, per_combo=6)
        candidates = read_jsonl(preview / "manifest.jsonl")
        # Pick different exclusion mechanisms within the same combo where possible.
        selected = [row for row in candidates if row["combo"] == "center+scratch"]
        other_manifest = self.root / "data" / "other-prepared" / "manifest.jsonl"
        write_jsonl(other_manifest, selected[:1])
        run = self.root / "runs" / "decision"
        write_json(run / "config.json", {"model": "gemini-decision-1", "selected_ids": [selected[1]["id"]]})
        write_json(run / "errors.json", [{"id": selected[2]["id"], "error": "attempted call"}])
        write_json(run / "raw" / f"{selected[3]['id']}.json", {"answers": {}})
        write_jsonl(run / "predictions.jsonl", [{"id": selected[4]["id"], "backend": "gemini-decision-1"}])
        report = self.prepare()
        fresh = read_jsonl(self.out / "manifest.jsonl")
        self.assertFalse({row["id"] for row in fresh} & {row["id"] for row in selected[:5]})
        self.assertEqual(report["audit_snapshot"]["api_exposed_or_configured_unique_ids"], 4)
        self.assertTrue(report["all_audited_overlaps_zero"])

    def test_insufficient_remaining_mixed_pool_fails_and_does_not_reuse_published_test(self):
        with self.assertRaisesRegex(ValueError, "Insufficient unused original-test"):
            self.prepare(per_combo=11)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.reference.read_bytes(), self.reference_bytes)


if __name__ == "__main__":
    unittest.main()
