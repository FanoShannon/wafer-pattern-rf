"""Test preprocessing invariants and optionally audit the prepared public dataset."""
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from waferbench.common import LABELS, ROOT, read_jsonl
from waferbench.data import dihedral_hash, render_map


class DataTransformTests(unittest.TestCase):
    def test_square_wafer_rotations_and_mirrors_share_one_group(self):
        wafer = np.array([[0, 0, 1, 0], [1, 2, 1, 1],
                          [1, 1, 2, 1], [0, 1, 1, 0]], dtype=np.uint8)
        expected = dihedral_hash(wafer)
        for rotation in range(4):
            for mirrored in (False, True):
                view = np.rot90(wafer, rotation)
                if mirrored:
                    view = np.fliplr(view)
                with self.subTest(rotation=rotation, mirrored=mirrored):
                    self.assertEqual(dihedral_hash(view), expected)
        changed = wafer.copy()
        changed[0, 2] = 2
        self.assertNotEqual(dihedral_hash(changed), expected)

    def test_render_has_only_legend_colors_and_preserves_die_blocks(self):
        wafer = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.uint8)
        palettes = {
            "cyan": [(255, 255, 255), (0, 224, 229), (188, 0, 239)],
            "mono": [(255, 255, 255), (216, 216, 216), (30, 30, 30)],
        }
        with tempfile.TemporaryDirectory(prefix="wafer-render-test-") as folder:
            for name, colors in palettes.items():
                path = Path(folder) / f"secret_label_should_never_be_drawn_{name}.png"
                render_map(wafer, path, palette=name, size=12)
                with Image.open(path) as image:
                    pixels = np.asarray(image)
                    self.assertEqual(image.size, (12, 12))
                    self.assertEqual(image.mode, "RGB")
                self.assertEqual(set(map(tuple, pixels.reshape(-1, 3))), set(colors))
                for row in range(3):
                    for col in range(3):
                        expected = np.asarray(colors[wafer[row, col]], dtype=np.uint8)
                        self.assertTrue(np.all(pixels[row*4:(row+1)*4, col*4:(col+1)*4] == expected))


_MANIFEST = ROOT / "data/mixedwm38/manifest.jsonl"
_PILOT = ROOT / "data/mixedwm38/pilot.jsonl"
_SOURCE = ROOT / "data_sources/Wafer_Map_Datasets.npz"


@unittest.skipUnless(_MANIFEST.exists() and _PILOT.exists() and _SOURCE.exists(),
                     "Optional real-data audit requires prepared MixedWM38 and original NPZ")
class PreparedMixedWM38AuditTests(unittest.TestCase):
    def test_source_alignment_quarantine_split_groups_and_balanced_pilot(self):
        rows, pilot = read_jsonl(_MANIFEST), read_jsonl(_PILOT)
        with np.load(_SOURCE, allow_pickle=False) as archive:
            source_maps, source_targets = archive["arr_0"], archive["arr_1"]
        seen_ids, seen_groups = set(), set()
        split_groups = defaultdict(set)
        for row in rows:
            idx = row["source_index"]
            wafer = np.load(ROOT / row["map_path"], allow_pickle=False)
            self.assertNotIn(row["id"], seen_ids)
            self.assertNotIn(row["group_id"], seen_groups)
            seen_ids.add(row["id"])
            seen_groups.add(row["group_id"])
            self.assertTrue(np.isin(wafer, [0, 1, 2]).all(), row["id"])
            np.testing.assert_array_equal(wafer, source_maps[idx], err_msg=row["id"])
            self.assertEqual(row["labels"], [name for name, bit in zip(LABELS, source_targets[idx]) if bit])
            self.assertEqual(row["group_id"], dihedral_hash(wafer))
            split_groups[row["split"]].add(row["group_id"])
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            self.assertFalse(split_groups[left] & split_groups[right])
        test_ids = {row["id"] for row in rows if row["split"] == "test"}
        self.assertTrue({row["id"] for row in pilot}.issubset(test_ids))
        self.assertEqual(len({row["id"] for row in pilot}), len(pilot))
        # The published pilot contract deliberately has five cases per combination.
        self.assertEqual(Counter(Counter(row["combo"] for row in pilot).values()), {5: 38})


if __name__ == "__main__":
    unittest.main()
