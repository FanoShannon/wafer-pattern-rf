"""Check augmentation split isolation, geometry, signals and honest shortfalls."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from waferbench.augmentation import _exclusive_statistics, _soft_structure, generate_composites
from waferbench.common import LABELS
from waferbench.data import dihedral_hash


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records = []
        yy, xx = np.indices((52, 52))
        self.active = (xx - 25.5)**2 + (yy - 25.5)**2 < 24**2
        rng = np.random.default_rng(35)
        patterns = {
            "center": (xx-25.5)**2 + (yy-25.5)**2 < 6**2,
            "edge_loc": (xx-43)**2 + (yy-25.5)**2 < 5**2,
            "loc": (xx-16)**2 + (yy-18)**2 < 4**2,
            "scratch": (yy == 28) & (xx >= 6) & (xx <= 45),
        }
        for label, mask in patterns.items():
            for number in range(3):
                wafer = self.active.astype(np.uint8)
                failed = (mask | (rng.random((52, 52)) < .025)) & self.active
                wafer[failed] = 2
                self.records.append(self.row(f"{label}-{number}", [label], wafer))
        for number in range(2):
            wafer = self.active.astype(np.uint8)
            wafer[(rng.random((52,52)) < .025) & self.active] = 2
            self.records.append(self.row(f"normal-{number}", [], wafer))
        # Vocabulary records do not provide pixels and therefore need no map file.
        for labels in (["center", "scratch"], ["center", "loc", "scratch"],
                       ["center", "edge_loc", "loc", "scratch"]):
            self.records.append({"id": "vocab-" + "+".join(labels), "labels": labels,
                                 "split": "train", "group_id": "vocab-group-" + "+".join(labels),
                                 "source": "fixture"})

    def tearDown(self):
        self.temp.cleanup()

    def row(self, item_id, labels, wafer):
        path = self.root / f"{item_id}.npy"
        np.save(path, wafer, allow_pickle=False)
        return {"id": item_id, "labels": labels, "split": "train", "source": "fixture",
                "group_id": dihedral_hash(wafer), "map_path": str(path)}

    def test_rejects_nontraining_parent_or_vocabulary_before_loading_maps(self):
        for split in ("val", "test"):
            records = [*self.records, {"id": "leak", "split": split}]
            with self.assertRaisesRegex(ValueError, "train records only"):
                generate_composites(records, n_per_combo=1)

    def test_output_geometry_targets_provenance_and_determinism(self):
        first = generate_composites(self.records, n_per_combo=3, seed=193, background_rate=.025)
        second = generate_composites(self.records, n_per_combo=3, seed=193, background_rate=.025)
        maps, targets, provenance, audit = first
        np.testing.assert_array_equal(maps, second[0])
        np.testing.assert_array_equal(targets, second[1])
        self.assertEqual(provenance, second[2])
        self.assertEqual(maps.shape, (9, 52, 52))
        self.assertEqual(targets.shape, (9, 8))
        self.assertEqual(maps.dtype, np.uint8)
        self.assertTrue(np.isin(maps, [0,1,2]).all())
        self.assertEqual(len({dihedral_hash(w) for w in maps}), len(maps))
        allowed = {tuple(sorted(r["labels"])) for r in self.records if 2 <= len(r["labels"]) <= 4}
        source_ids = {r["id"] for r in self.records if len(r["labels"]) == 1}
        for wafer, target, record in zip(maps, targets, provenance):
            self.assertIn(tuple(sorted(record["labels"])), allowed)
            self.assertEqual(set(np.asarray(LABELS)[target.astype(bool)]), set(record["labels"]))
            self.assertEqual(record["split"], "train")
            self.assertEqual(record["new_shared_background_dies"], record["output_fail_dies"] - record["heuristic_signal_union_dies"])
            for parent in record["parents"]:
                self.assertIn(parent["source_id"], source_ids)
                self.assertEqual(parent["source_split"], "train")
                self.assertGreaterEqual(parent["exclusive_signal_fraction"], .05)
                self.assertGreaterEqual(parent["exclusive_signal_dies"], 3)
            self.assertEqual(int((wafer > 0).sum()), record["wafer_active_dies"])
        self.assertEqual(audit["total_shortfall"], 0)
        self.assertNotIn("random", audit["unique_single_parent_counts"])
        self.assertNotIn("near_full", audit["unique_single_parent_counts"])
        json.dumps(audit, allow_nan=False)
        self.assertEqual(len(audit["signal_qa_examples"]), 4)

    def test_raw_union_control_is_exact_union_on_shared_geometry(self):
        maps, _, provenance, audit = generate_composites(self.records, n_per_combo=2,
                                                       mode="naive_union", seed=57)
        for wafer, row in zip(maps, provenance):
            expected_active, expected_fail = None, np.zeros((52,52), dtype=bool)
            for parent in row["parents"]:
                original = np.load(parent["source_map_path"])
                code = parent["d4_code"]
                transformed = np.rot90(np.fliplr(original) if code >= 4 else original, code % 4)
                active = transformed > 0
                expected_active = active if expected_active is None else expected_active & active
                expected_fail |= transformed == 2
            np.testing.assert_array_equal(wafer > 0, expected_active)
            np.testing.assert_array_equal(wafer == 2, expected_fail & expected_active)
            self.assertEqual(row["new_shared_background_dies"], 0)
        self.assertEqual(audit["background_estimate"]["method"], "median_fail_density_of_training_normal_maps")

    def test_empty_parent_shortfall_is_reported_without_padding(self):
        empty = self.row("empty-center", ["center"], self.active.astype(np.uint8))
        records = [empty, next(r for r in self.records if r["labels"] == ["scratch"]),
                   {"id": "cs-vocab", "labels": ["center", "scratch"], "split": "train", "group_id": "cs-vocab"}]
        maps, targets, provenance, audit = generate_composites(records, n_per_combo=4)
        self.assertEqual(maps.shape, (0,52,52))
        self.assertEqual(targets.shape, (0,8))
        self.assertEqual(provenance, [])
        self.assertEqual(audit["total_shortfall"], 4)
        self.assertEqual(audit["invariants"]["duplicates_padded_to_quota"], 0)
        self.assertEqual(audit["combinations"][0]["missing_parent_labels"], ["center"])

    def test_exclusive_support_detects_complete_occlusion(self):
        same = np.zeros((52,52), dtype=bool)
        same[10:20,10:20] = True
        self.assertTrue(all(stat["exclusive_signal_fraction"] == 0 for stat in _exclusive_statistics([same, same])))

    def test_nested_parent_signal_is_rejected_and_attempts_are_bounded(self):
        broad = np.ones((52,52), dtype=np.uint8)
        broad[10:42,10:42] = 2
        nested = np.ones((52,52), dtype=np.uint8)
        nested[24:28,24:28] = 2
        records = [self.row("broad", ["center"], broad), self.row("nested", ["loc"], nested),
                   {"id": "nested-vocab", "labels": ["center", "loc"],
                    "split": "train", "group_id": "nested-vocab"}]
        maps, _, _, audit = generate_composites(records, n_per_combo=1, seed=3,
                                               background_rate=0, max_attempts_per_sample=3)
        self.assertEqual(len(maps), 0)
        self.assertEqual(audit["total_shortfall"], 1)
        self.assertEqual(audit["candidate_attempts"], 3)
        self.assertEqual(audit["rejection_counts"]["parent_signal_occluded"], 3)

    def test_modes_share_parent_transform_candidates_with_independent_noise_rng(self):
        raw = generate_composites(self.records, n_per_combo=2, seed=99, mode="naive_union")
        structured = generate_composites(self.records, n_per_combo=2, seed=99, mode="structured_mix")
        self.assertEqual([r["parent_transform_signature"] for r in raw[2]],
                         [r["parent_transform_signature"] for r in structured[2]])
        self.assertEqual([r["heuristic_signal_union_dies"] for r in raw[2]],
                         [r["heuristic_signal_union_dies"] for r in structured[2]])

    def test_directional_evidence_keeps_thin_trace_probabilistic_support(self):
        wafer = np.ones((52,52), dtype=np.uint8)
        wafer[25,8:44] = 2
        wafer[4,4] = 2
        score = _soft_structure(wafer, .1, .05)
        self.assertGreater(float(score[25,12:40].mean()), .99)
        self.assertLess(float(score[4,4]), .1)
        self.assertTrue(np.all(score[wafer != 2] == 0))

    def test_budget_limits_and_unobserved_combinations(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            generate_composites(self.records, n_per_combo=1400)
        with self.assertRaisesRegex(ValueError, "fewer than 4000"):
            generate_composites(self.records, n_per_combo=1, max_total_attempts=4000)
        singles = [r for r in self.records if len(r["labels"]) <= 1]
        maps, targets, _, audit = generate_composites(singles, n_per_combo=1)
        self.assertEqual(maps.shape, (0,52,52))
        self.assertEqual(targets.shape, (0,8))
        self.assertEqual(audit["requested_total"], 0)


if __name__ == "__main__":
    unittest.main()
