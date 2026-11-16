"""Validate the versioned feature contract and meaningful geometric behavior."""
import unittest

import numpy as np

from waferbench.advanced_features import (
    advanced_features,feature_names,FEATURE_COUNT,FEATURE_GROUPS,
)
from waferbench.baseline import spatial_features


def disk(size=61):
    yy,xx = np.indices((size,size))
    return np.where((yy-(size-1)/2)**2+(xx-(size-1)/2)**2<=((size-3)/2)**2,1,0).astype(np.uint8)


class AdvancedFeatureTests(unittest.TestCase):
    def test_versioned_dimension_names_finite_and_original_features(self):
        names = feature_names()
        self.assertEqual(len(names),FEATURE_COUNT)
        self.assertEqual(len(set(names)),FEATURE_COUNT)
        self.assertEqual(FEATURE_GROUPS["component_shape"][1],FEATURE_COUNT)
        for state in (1,2):
            wafer = disk()
            wafer[wafer>0] = state
            original = wafer.copy()
            features = advanced_features(wafer)
            self.assertEqual(features.shape,(502,))
            self.assertEqual(features.dtype,np.float32)
            self.assertTrue(np.isfinite(features).all())
            np.testing.assert_array_equal(features[:168],spatial_features(wafer))
            np.testing.assert_array_equal(wafer,original)

    def test_added_channels_ignore_outside_padding(self):
        wafer = disk(31)
        wafer[10:15,13:17] = 2
        wafer[18,8:23] = 2
        original = advanced_features(wafer)
        padded = advanced_features(np.pad(wafer,((3,5),(6,2)),constant_values=0))
        np.testing.assert_allclose(original[168:],padded[168:],atol=1e-6,rtol=1e-6)
        # Existing area/whole-image geometry ratio intentionally records padding.
        self.assertNotEqual(original[1],padded[1])

    def test_line_evidence_exceeds_equal_area_compact_blob_and_tracks_rotation(self):
        line = disk()
        line[30,6:55] = 2  # 49 failed dies.
        blob = disk()
        blob[27:34,27:34] = 2  # Exactly the same number of failed dies.
        names = feature_names()
        peak_indices = [i for i,name in enumerate(names) if name.startswith("line_normal_") and name.endswith("peak_z")]
        horizontal = advanced_features(line)[peak_indices]
        compact = advanced_features(blob)[peak_indices]
        vertical = advanced_features(np.rot90(line))[peak_indices]
        self.assertGreater(float(horizontal.max()),float(compact.max())*1.5)
        self.assertEqual(int(horizontal.argmax()),9)  # Normal 90 degrees.
        self.assertEqual(int(vertical.argmax()),0)  # Normal 0 degrees.
        self.assertAlmostEqual(float(horizontal.max()),float(vertical.max()),places=5)

    def test_joint_polar_channels_distinguish_rim_arc_from_full_ring(self):
        yy,xx = np.indices((61,61))
        radius = np.hypot(xx-30,yy-30)
        ring,arc = disk(),disk()
        ring[(radius>=25)&(ring>0)] = 2
        arc[(radius>=25)&(xx>30)&(yy>30)&(arc>0)] = 2
        start,stop = FEATURE_GROUPS["polar_joint"]
        ring_density,arc_density = advanced_features(ring)[start:stop],advanced_features(arc)[start:stop]
        # A rim ring occupies many angular cells, while an arc remains localized.
        self.assertGreater(int((ring_density>.5).sum()),3*int((arc_density>.5).sum()))

    def test_empty_footprint_and_invalid_die_codes_raise(self):
        for wafer in (np.zeros((5,5)),np.full((5,5),3),np.ones((5,5,1))):
            with self.assertRaises(ValueError):
                advanced_features(wafer)


if __name__=="__main__":
    unittest.main()
