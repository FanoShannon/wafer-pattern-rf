"""Run the full ablation workflow on temporary procedural maps, never benchmarks.

The independent audit fixture below only tests workflow wiring and split
invariants. Its toy labels/scores are not evidence about real wafer accuracy.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from examples.smoke_demo import make_toy_fixture
from waferbench import augmentation_study as study
from waferbench.common import read_jsonl,write_jsonl


class StudySmokeTests(unittest.TestCase):
    def test_full_study_without_historical_runs_freezes_before_paired_audit(self):
        with tempfile.TemporaryDirectory(prefix="wafer-study-workflow-only-") as directory:
            root=Path(directory)
            manifest=make_toy_fixture(root/"data/mixedwm38",seed=721)
            rows=read_jsonl(manifest)
            self.assertEqual(len(rows),40)
            self.assertTrue(all(r["source"]=="procedural_smoke_demo_only" for r in rows))
            # A second seed supplies independent procedural pixels. Rename the
            # audit ID because the demo intentionally reuses its class/split IDs.
            other=read_jsonl(make_toy_fixture(root/"other_procedural_source",seed=1841))
            fresh=[{**r,"id":"workflow-audit-"+r["id"],
                    "audit_role":"independent_procedural_workflow_fixture_only"}
                   for r in other if r["split"]=="test" and len(r["labels"])>=2]
            self.assertEqual(len(fresh),1)
            self.assertFalse({r["group_id"] for r in fresh}&{r["group_id"] for r in rows})
            self.assertTrue(all(r["source"]=="procedural_smoke_demo_only" for r in fresh))
            write_jsonl(root/"data/mixedwm38_fresh_mixed/manifest.jsonl",fresh)
            historical=root/"runs/mixedwm38/rf_multi/val_predictions.jsonl"
            self.assertFalse(historical.exists())
            out=root/"runs/study"
            frozen_path=out/"selection_frozen.json"
            prediction_events=[]
            frozen_bytes=[]
            original_records=study.records_from_probabilities

            def observe_predictions(selected,probabilities,arm,seed):
                kind="audit" if selected[0]["id"].startswith("workflow-audit-") else selected[0]["split"]
                if kind=="val":
                    self.assertFalse(frozen_path.exists(),"Selection must use validation before freezing")
                else:
                    self.assertTrue(frozen_path.exists(),"Old-test/audit inference preceded validation freeze")
                    if not frozen_bytes:
                        frozen_bytes.append(frozen_path.read_bytes())
                    self.assertEqual(frozen_path.read_bytes(),frozen_bytes[0],"Selection changed after freeze")
                prediction_events.append((kind,arm,seed))
                return original_records(selected,probabilities,arm,seed)

            # Only reduce tree count/thread overhead for test speed. Feature
            # extraction, all ten arms, selection, bootstrap, audit and reporting
            # run normally; the public default of 300 trees remains untouched.
            small_parameters={**study.PARAMETERS,"n_estimators":8,"n_jobs":1}
            with patch.object(study,"ROOT",root),patch.object(study,"PARAMETERS",small_parameters), \
                 patch.object(study,"records_from_probabilities",side_effect=observe_predictions), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary=study.run_study(out=out,n_per_combo=1)
            self.assertEqual(study.PARAMETERS["n_estimators"],300)
            self.assertFalse(historical.exists())
            reproduction=json.loads((out/"baseline_reproduction.json").read_text(encoding="utf-8"))
            self.assertEqual(reproduction["status"],"fresh_clone_no_historical_run")
            expected_arms={f"{feature}_{augmentation}" for feature in ["base","advanced"]
                           for augmentation in ["none","d4_control","naive_union","structured_mix"]}
            expected_arms|={"advanced_single_only","advanced_single_plus_structured"}
            self.assertEqual({arm["arm"] for arm in summary["arms"]},expected_arms)
            self.assertEqual(len(summary["arms"]),10)
            self.assertEqual(sum(r["candidate_for_primary_selection"] for r in summary["arms"]),8)
            self.assertEqual(sum(kind=="val" for kind,_,_ in prediction_events),10)
            self.assertEqual(sum(kind=="test" for kind,_,_ in prediction_events),10)
            self.assertEqual(frozen_path.read_bytes(),frozen_bytes[0])
            self.assertEqual(summary["selection"]["selected_arm"],summary["selection"]["validation_ranking"][0]["arm"])
            chosen=summary["selection"]["selected_arm"]
            expected_audit_arms={"base_none",chosen}
            self.assertEqual({r["arm"] for r in summary["fresh_audit"]},expected_audit_arms)
            self.assertEqual({arm for kind,arm,_ in prediction_events if kind=="audit"},expected_audit_arms)
            for arm in summary["fresh_audit"]:
                self.assertEqual([r["seed"] for r in arm["seeds"]],study.REPLICATION_SEEDS)
                self.assertTrue(all(r["metrics"]["n"]==len(fresh) for r in arm["seeds"]))
            expected_files={out/arm/f"fresh_predictions_seed_{seed}.jsonl"
                            for arm in expected_audit_arms for seed in study.REPLICATION_SEEDS}
            self.assertEqual(set(out.glob("*/fresh_predictions_seed_*.jsonl")),expected_files)
            for path in expected_files:
                predictions=read_jsonl(path)
                self.assertEqual([r["id"] for r in predictions],[r["id"] for r in fresh])
            overlap=json.loads((out/"fresh_augmentation_overlap_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(overlap["id_overlap_with_reference"],0)
            self.assertTrue(all(value==0 for value in overlap["generated_d4_overlap"].values()))
            receipt=json.loads((out/"fresh_evaluation_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(set(receipt["evaluated_arms"]),expected_audit_arms)
            self.assertEqual(receipt["training_seeds"],study.REPLICATION_SEEDS)
            self.assertEqual(receipt["n"],len(fresh))
            for filename in ["report.md","summary.json","ablation.png","augmentation_preview.png",
                             "validation_group_importance.json","selected_model.joblib","feature_names.json"]:
                self.assertTrue((out/filename).is_file(),filename)
            bundle=joblib.load(out/"selected_model.joblib")
            self.assertEqual(bundle["task"],"multi")
            self.assertEqual(bundle["arm"],chosen)
            self.assertEqual(bundle["n_features"],bundle["model"].n_features_in_)
            self.assertEqual(bundle["thresholds"],[.5]*8)
            names=json.loads((out/"feature_names.json").read_text(encoding="utf-8"))
            self.assertEqual(len(names),bundle["n_features"])
            self.assertEqual(summary["config"]["parameters"]["n_estimators"],8)
            self.assertEqual(summary["config"]["external_api_calls"],0)


if __name__=="__main__":
    unittest.main()
