"""Checks for calibration semantics, normal labels, and malformed API output."""
import unittest

from waferbench.metrics import evaluate_predictions


class MetricsTests(unittest.TestCase):
    def test_single_normal_and_tied_risk_groups(self):
        rows = [
            {"id": "a", "labels": [], "predicted_labels": [], "probabilities": {"normal": .8, "scratch": .2}},
            {"id": "b", "labels": ["scratch"], "predicted_labels": [], "probabilities": {"normal": .8, "scratch": .2}},
        ]
        result = evaluate_predictions(rows, ["normal", "scratch"], "single")
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["brier_score"], .68)
        self.assertAlmostEqual(result["ece_10_bins"], .3)
        self.assertEqual(len(result["risk_coverage"]), 1)
        self.assertEqual(result["risk_coverage"][0]["risk"], .5)

    def test_multilabel_uses_independent_probabilities_and_empty_normal(self):
        rows = [
            {"id": "a", "labels": ["center", "scratch"], "predicted_labels": ["center"], "probabilities": {"center": .9, "scratch": .4}},
            {"id": "b", "labels": [], "predicted_labels": [], "probabilities": {"center": .1, "scratch": .1}},
        ]
        result = evaluate_predictions(rows, ["center", "scratch"], "multi")
        self.assertAlmostEqual(result["micro_f1"], 2/3)
        self.assertEqual(result["hamming_loss"], .25)
        self.assertEqual(result["exact_match_accuracy"], .5)
        self.assertEqual(result["mixed_only"]["n"], 1)
        self.assertAlmostEqual(result["samples_f1"], 5/6)

    def test_missing_probability_and_failed_request_raise(self):
        good = {"id": "a", "labels": [], "predicted_labels": [], "probabilities": {"normal": .8, "scratch": .2}}
        for change in [{"probabilities": {"normal": 1}}, {"error": "timeout"}, {"probabilities": {"normal": float("nan"), "scratch": .2}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_predictions([{**good, **change}], ["normal", "scratch"], "single")
        with self.assertRaises(ValueError):
            evaluate_predictions([good, good], ["normal", "scratch"], "single")


if __name__ == "__main__":
    unittest.main()
