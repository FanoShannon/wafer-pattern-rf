"""Fresh-checkout CLI smoke tests using only generated, temporary local maps."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("rf_local_smoke_demo", REPOSITORY_ROOT / "examples" / "smoke_demo.py")
_DEMO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEMO)


class CLISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="wafer-rf-cli-test-")
        cls.root = Path(cls.temporary.name)
        cls.manifest = _DEMO.make_toy_fixture(cls.root / "fixture")
        cls.rows = [json.loads(line) for line in cls.manifest.read_text(encoding="utf-8").splitlines()]
        cls.empty_cwd = cls.root / "unrelated-empty-working-directory"
        cls.empty_cwd.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_cli(self, *arguments):
        # Isolated Python ignores PYTHONPATH/user-site/cwd, and imports only this
        # checkout plus installed dependencies. No original research tree is used.
        # The socket guard makes accidental API/download attempts fail the test.
        bootstrap = (
            "import runpy, socket, sys\n"
            "sys.path.insert(0, sys.argv.pop(1))\n"
            "def no_network(*args, **kwargs):\n"
            "    raise AssertionError('The RF smoke test forbids network calls')\n"
            "socket.socket.connect = no_network\n"
            "socket.create_connection = no_network\n"
            "runpy.run_module('waferbench', run_name='__main__')\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", bootstrap, str(REPOSITORY_ROOT), *map(str, arguments)],
            cwd=self.empty_cwd, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0,
                         f"CLI failed for {arguments!r}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return result

    def test_fixture_has_true_unique_d4_groups_and_all_split_classes(self):
        from waferbench.common import SINGLE_LABELS
        from waferbench.data import dihedral_hash

        self.assertEqual(len(self.rows), 40)
        self.assertEqual(Counter(row["split"] for row in self.rows), {"train": 20, "val": 10, "test": 10})
        seen = set()
        for row in self.rows:
            wafer = np.load(row["map_path"], allow_pickle=False)
            self.assertEqual(wafer.shape, (52, 52))
            self.assertTrue(np.isin(wafer, [0, 1, 2]).all())
            self.assertEqual(dihedral_hash(wafer), row["group_id"])
            self.assertNotIn(row["group_id"], seen)
            seen.add(row["group_id"])
            self.assertEqual(row["source"], "procedural_smoke_demo_only")
        for split, expected_count in [("train", 2), ("val", 1), ("test", 1)]:
            counts = Counter(row["combo"] for row in self.rows if row["split"] == split)
            for category in SINGLE_LABELS:
                self.assertEqual(counts[category], expected_count)
        self.assertEqual(len(seen), 40)

    def test_cli_single_and_multilabel_end_to_end_from_empty_cwd(self):
        from waferbench.common import LABELS, SINGLE_LABELS

        for task, features, names, expected_n, feature_rows, feature_count, split_sizes in [
            ("single", "base", SINGLE_LABELS, 9, 36, 168, {"train": 18, "val": 9, "test": 9}),
            ("multi", "base", LABELS, 10, 40, 168, {"train": 20, "val": 10, "test": 10}),
            ("multi", "advanced", LABELS, 10, 40, 502, {"train": 20, "val": 10, "test": 10}),
        ]:
            with self.subTest(task=task, features=features):
                out = self.root / "results" / f"{task}-{features}"
                self.run_cli("baseline", "--manifest", self.manifest, "--out", out,
                             "--task", task, "--seed", 721, "--features", features)
                metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
                predictions = [json.loads(line) for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                validation = [json.loads(line) for line in (out / "val_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                expected_ids = {row["id"] for row in self.rows if row["split"] == "test"
                                and (task == "multi" or len(row["labels"]) <= 1)}
                self.assertEqual(len(predictions), expected_n)
                self.assertEqual({row["id"] for row in predictions}, expected_ids)
                self.assertEqual(len(validation), split_sizes["val"])
                self.assertTrue(all(row["split"] == "val" for row in validation))
                self.assertEqual(metrics["n"], expected_n)
                self.assertEqual(metrics["run"]["split_sizes"], split_sizes)
                self.assertEqual(metrics["run"]["n_features"], feature_count)
                self.assertTrue((out / "model.joblib").is_file())
                metadata = json.loads((out / "model_metadata.json").read_text(encoding="utf-8"))
                self.assertIsInstance(metadata, dict)
                cache_paths = list(out.glob("features_*.npz"))
                self.assertEqual(len(cache_paths), 1)
                with np.load(cache_paths[0], allow_pickle=False) as cache:
                    self.assertEqual(cache["features"].shape, (feature_rows, feature_count))
                    self.assertTrue(np.isfinite(cache["features"]).all())
                for row in predictions:
                    self.assertEqual(row["split"], "test")
                    self.assertEqual(set(row["probabilities"]), set(names))
                    probabilities = np.asarray([row["probabilities"][name] for name in names])
                    self.assertTrue(np.isfinite(probabilities).all())
                    self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                    if task == "single":
                        self.assertAlmostEqual(probabilities.sum(), 1)
                        self.assertLessEqual(len(row["predicted_labels"]), 1)
                    else:
                        self.assertEqual(set(metrics["thresholds"]), set(LABELS))
                        expected = [name for name in LABELS
                                    if row["probabilities"][name] >= metrics["thresholds"][name]]
                        self.assertEqual(row["predicted_labels"], expected)
                evaluated_path = self.root / "results" / f"{task}-{features}-reevaluated.json"
                self.run_cli("evaluate", "--predictions", out / "predictions.jsonl",
                             "--task", task, "--out", evaluated_path)
                evaluated = json.loads(evaluated_path.read_text(encoding="utf-8"))
                self.assertEqual(evaluated["n"], expected_n)
                self.assertEqual(evaluated["labels"], names)
                first = predictions[0]
                original = next(row for row in self.rows if row["id"] == first["id"])
                predicted_path = self.root / "results" / f"{task}-{features}-reloaded.json"
                self.run_cli("predict", "--model", out / "model.joblib", "--map", original["map_path"],
                             "--out", predicted_path)
                reloaded = json.loads(predicted_path.read_text(encoding="utf-8"))
                self.assertEqual(reloaded["predicted_labels"], first["predicted_labels"])
                self.assertEqual(set(reloaded["probabilities"]), set(names))
                np.testing.assert_allclose(
                    [reloaded["probabilities"][name] for name in names],
                    [first["probabilities"][name] for name in names], rtol=0, atol=1e-12)
                # Toy scores have no benchmark target; this checks serialization,
                # feature extraction, fitting, validation decoding and test I/O.

    def test_advanced_feature_export_remains_available_without_real_data(self):
        from waferbench.advanced_features import advanced_features, feature_names
        from waferbench.baseline import spatial_features

        wafer = np.load(self.rows[0]["map_path"], allow_pickle=False)
        values = advanced_features(wafer)
        self.assertEqual(values.shape, (502,))
        self.assertEqual(len(feature_names()), 502)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[:168], spatial_features(wafer))


if __name__ == "__main__":
    unittest.main()
