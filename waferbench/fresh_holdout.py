"""Prepare an unscored mixed-only holdout from unused original test-pool groups.

This audits unused maps from the same public archive, not newly collected fab
data. Source labels were already available to the original splitter. Preparation
neither trains a model nor computes predictions or evaluation metrics.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import LABELS, ROOT, read_jsonl, resolve_path, write_json, write_jsonl
from .data import dihedral_hash, sha256_file

CAPS = {"train": 200, "val": 20, "test": 20}
IDENTITY_FIELDS = ("id", "source_index", "group_id", "split", "labels", "combo", "duplicate_multiplicity")


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _replay(maps: np.ndarray, targets: np.ndarray, seed: int) -> tuple[list[dict], dict, dict, dict]:
    """Replay prepare_mixed's exact sorted-group RNG sequence and capped split."""
    if maps.ndim != 3 or targets.shape != (len(maps), len(LABELS)):
        raise ValueError(f"Unexpected MixedWM38 shapes: {maps.shape}, {targets.shape}")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Source targets must be the canonical eight binary components")
    valid = np.isin(maps, [0, 1, 2]).all(axis=(1, 2))
    groups = defaultdict(list)
    index_groups = {}
    for index in np.flatnonzero(valid):
        group = dihedral_hash(maps[index].astype(np.uint8))
        groups[group].append(int(index))
        index_groups[int(index)] = group
    by_bits = defaultdict(list)
    conflicts = 0
    for group, indices in groups.items():
        options = {tuple(int(value) for value in targets[index]) for index in indices}
        if len(options) != 1:
            conflicts += 1
            continue
        by_bits[next(iter(options))].append((group, indices[0], len(indices)))
    rng = np.random.default_rng(seed)
    reference, original_test_pools = [], {}
    for bits, members in sorted(by_bits.items()):
        members.sort()
        rng.shuffle(members)
        if len(members) < 7:
            raise ValueError(f"Insufficient source groups for original three-way split: {bits}")
        n_test = max(1, round(len(members) * .15))
        n_val = max(1, round(len(members) * .15))
        partitions = {"test": members[:n_test], "val": members[n_test:n_test+n_val],
                      "train": members[n_test+n_val:]}
        labels = [name for name, bit in zip(LABELS, bits) if bit]
        combo = "+".join(labels) or "normal"

        def row(member, split):
            group, index, multiplicity = member
            return {"id": f"mixedwm38-{index:05d}", "source": "MixedWM38", "source_index": index,
                    "group_id": group, "split": split, "labels": labels.copy(), "combo": combo,
                    "duplicate_multiplicity": multiplicity}

        original_test_pools[combo] = [row(member, "test") for member in partitions["test"]]
        for split, pool in partitions.items():
            reference.extend(row(member, split) for member in pool[:CAPS[split]])
    reference.sort(key=lambda item: item["id"])
    diagnostics = {"raw_maps": len(maps), "valid_code_maps": int(valid.sum()),
                   "quarantined_invalid_code_maps": int((~valid).sum()),
                   "dihedral_groups_after_code_filter": len(groups),
                   "quarantined_label_conflict_groups": conflicts,
                   "source_combinations_after_quarantine": len(by_bits)}
    return reference, original_test_pools, index_groups, diagnostics


def _verify_reference(replayed: list[dict], reference: list[dict]) -> None:
    if len(reference) != len(replayed):
        raise ValueError("Reference manifest row count differs from exact 200/20/20 split replay")
    observed = {}
    for row in reference:
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier in observed:
            raise ValueError("Reference manifest contains missing or duplicate IDs")
        observed[identifier] = row
    if set(observed) != {row["id"] for row in replayed}:
        raise ValueError("Reference source-index selection differs from exact split replay")
    for expected in replayed:
        actual = observed[expected["id"]]
        if actual.get("source") != "MixedWM38":
            raise ValueError("Reference source must be MixedWM38")
        if any(actual.get(field) != expected[field] for field in IDENTITY_FIELDS):
            raise ValueError(f"Reference source index/group/split/labels differ at {expected['id']}")


def _collect_exposure(reference_manifest: Path, out: Path, index_groups: dict) -> dict:
    """Audit existing prepared manifests and model/API artifacts, never secrets."""
    dataset_ids, dataset_groups = set(), set()
    model_ids, model_groups = set(), set()
    api_ids, api_groups = set(), set()
    manifests, model_files, api_files = [], [], []

    def add(identifier, group, ids, groups):
        if isinstance(identifier, str) and identifier:
            ids.add(identifier)
            match = re.fullmatch(r"mixedwm38-(\d+)", identifier)
            if match and int(match.group(1)) in index_groups:
                groups.add(index_groups[int(match.group(1))])
        if isinstance(group, str) and group:
            groups.add(group)

    paths = set((ROOT / "data").rglob("manifest.jsonl")) if (ROOT / "data").exists() else set()
    paths.add(reference_manifest)
    for path in sorted(paths):
        path = path.resolve()
        if path == out / "manifest.jsonl" or out in path.parents:
            continue
        rows = read_jsonl(path)
        for row in rows:
            add(row.get("id"), row.get("group_id"), dataset_ids, dataset_groups)
        manifests.append({"path": _path_label(path), "sha256": sha256_file(path), "rows": len(rows)})

    run_root = ROOT / "runs"
    api_directories = set()
    if run_root.exists():
        for path in sorted(run_root.rglob("config.json")):
            config = json.loads(path.read_text(encoding="utf-8-sig"))
            is_api = (str(config.get("model", "")).startswith("gemini")
                      or "generativelanguage.googleapis.com" in str(config.get("endpoint", "")))
            if not is_api:
                continue
            api_directories.add(path.parent)
            selected = config.get("selected_ids", [])
            if not isinstance(selected, list):
                raise ValueError(f"Cannot audit API selected_ids in {_path_label(path)}")
            for identifier in selected:
                add(identifier, None, api_ids, api_groups)
            api_files.append({"path": _path_label(path), "sha256": sha256_file(path),
                              "selection_ids": len(selected), "kind": "configured_selection_conservative"})
        for path in sorted(run_root.rglob("*prediction*.jsonl")):
            rows = read_jsonl(path)
            for row in rows:
                add(row.get("id"), row.get("group_id"), model_ids, model_groups)
                if path.parent in api_directories or str(row.get("backend", "")).startswith("gemini"):
                    add(row.get("id"), row.get("group_id"), api_ids, api_groups)
            model_files.append({"path": _path_label(path), "sha256": sha256_file(path), "rows": len(rows)})
        for directory in sorted(api_directories):
            errors = directory / "errors.json"
            if errors.exists():
                content = json.loads(errors.read_text(encoding="utf-8-sig"))
                rows = content if isinstance(content, list) else content.get("errors", [])
                for row in rows:
                    add(row.get("id"), row.get("group_id"), api_ids, api_groups)
                api_files.append({"path": _path_label(errors), "sha256": sha256_file(errors),
                                  "rows": len(rows), "kind": "attempted_call_errors"})
            raw_files = sorted((directory / "raw").glob("*.json"))
            for path in raw_files:
                add(path.stem, None, api_ids, api_groups)
            if raw_files:
                api_files.append({"path": _path_label(directory / "raw"), "rows": len(raw_files),
                                  "kind": "raw_response_filenames"})
    return {"dataset_ids": dataset_ids, "dataset_groups": dataset_groups,
            "model_ids": model_ids, "model_groups": model_groups,
            "api_ids": api_ids, "api_groups": api_groups,
            "dataset_manifests": manifests, "model_prediction_files": model_files, "api_artifacts": api_files}


def prepare_fresh_mixed_holdout(
        source: Path = Path("data_sources/Wafer_Map_Datasets.npz"),
        reference_manifest: Path = Path("data/mixedwm38/manifest.jsonl"),
        out: Path = Path("data/mixedwm38_fresh_mixed"), seed: int = 42,
        per_combo: int = 20) -> dict:
    """Prepare unused mixed groups, after proving the original split is unchanged.

    Selection follows the original shuffled test-pool order after excluding all
    published/prepared and previously evaluated/configured IDs and D4 groups.
    Normal and single-pattern pools (including exhausted near-full) are skipped.
    """
    if isinstance(per_combo, bool) or not isinstance(per_combo, int) or per_combo <= 0:
        raise ValueError("per_combo must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    source, reference_manifest, out = (resolve_path(path).resolve() for path in [source, reference_manifest, out])
    if (out / "manifest.jsonl").exists() or (out / "report.json").exists():
        raise FileExistsError("Fresh holdout already exists; preserve it instead of selecting again")
    reference_bytes = reference_manifest.read_bytes()
    reference = [json.loads(line) for line in reference_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    source_hash = sha256_file(source)
    prior_report_path = reference_manifest.parent / "data_report.json"
    if prior_report_path.exists():
        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8-sig"))
        if prior_report.get("source_sha256") != source_hash:
            raise ValueError("Source archive SHA256 differs from the reference data report")
    with np.load(source, allow_pickle=False) as archive:
        maps, targets = archive["arr_0"], archive["arr_1"]
    replayed, pools, index_groups, diagnostics = _replay(maps, targets, seed)
    _verify_reference(replayed, reference)
    exposure = _collect_exposure(reference_manifest, out, index_groups)
    excluded_ids = exposure["dataset_ids"] | exposure["model_ids"] | exposure["api_ids"]
    excluded_groups = exposure["dataset_groups"] | exposure["model_groups"] | exposure["api_groups"]
    fresh, inventory = [], []
    for combo, pool in sorted(pools.items()):
        if len(pool[0]["labels"]) < 2:
            continue
        available = [row for row in pool if row["id"] not in excluded_ids and row["group_id"] not in excluded_groups]
        if len(available) < per_combo:
            raise ValueError(f"Insufficient unused original-test groups for {combo}: {len(available)} < {per_combo}")
        fresh.extend(available[:per_combo])
        inventory.append({"combo": combo, "original_test_pool_groups": len(pool),
                          "excluded_prepared_or_exposed_groups": len(pool)-len(available),
                          "unused_eligible_groups": len(available), "selected": per_combo})
    if not fresh:
        raise ValueError("Source has no eligible mixed-pattern combinations")
    fresh.sort(key=lambda row: row["id"])
    fresh_ids, fresh_groups = {row["id"] for row in fresh}, {row["group_id"] for row in fresh}
    if len(fresh_ids) != len(fresh) or len(fresh_groups) != len(fresh):
        raise ValueError("Fresh holdout IDs and D4 groups must each be unique")
    overlaps = {}
    for category in ["dataset", "model", "api"]:
        overlaps[f"existing_{category}_ids"] = len(fresh_ids & exposure[f"{category}_ids"])
        overlaps[f"existing_{category}_d4_groups"] = len(fresh_groups & exposure[f"{category}_groups"])
    for split in ["train", "val", "test"]:
        original = [row for row in reference if row["split"] == split]
        overlaps[f"reference_{split}_ids"] = len(fresh_ids & {row["id"] for row in original})
        overlaps[f"reference_{split}_d4_groups"] = len(fresh_groups & {row["group_id"] for row in original})
    if any(overlaps.values()):
        raise ValueError("Fresh holdout overlaps an existing dataset or previously exposed model/API input")
    if reference_manifest.read_bytes() != reference_bytes:
        raise ValueError("Reference manifest changed during preparation")
    report = {
        "dataset": "MixedWM38 fresh mixed-only audit holdout", "status": "prepared_unscored",
        "source_file": str(source), "source_sha256": source_hash, "raw_shape": list(maps.shape),
        "label_order": LABELS, "seed": seed, "per_combo": per_combo, "n": len(fresh),
        "mixed_combinations": len(inventory), "selected_split_counts": {"test": len(fresh)},
        "cardinality_counts": dict(sorted(Counter(len(row["labels"]) for row in fresh).items())),
        "reference_manifest": str(reference_manifest),
        "reference_manifest_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "reference_replay": {"exact_match": True, "identity_fields": list(IDENTITY_FIELDS),
                             "caps_per_combo": CAPS, "partition": "group-level 70/15/15, test/val first; original RNG sequence",
                             "verified_original_split_counts": dict(Counter(row["split"] for row in reference)),
                             "train_and_val_modified": False},
        "source_diagnostics": diagnostics, "selection_inventory": inventory,
        "overlap_counts": overlaps, "all_audited_overlaps_zero": True,
        "audit_snapshot": {"dataset_manifests": exposure["dataset_manifests"],
                           "model_prediction_files": exposure["model_prediction_files"],
                           "api_artifacts": exposure["api_artifacts"],
                           "existing_dataset_unique_ids": len(exposure["dataset_ids"]),
                           "prior_model_prediction_unique_ids": len(exposure["model_ids"]),
                           "api_exposed_or_configured_unique_ids": len(exposure["api_ids"]),
                           "policy": "Conservatively exclude all configured API selected_ids, successful predictions, error IDs and raw response IDs."},
        "selection_policy": "First eligible groups in the original shuffled test-pool order; mixed only; no model scores or errors used.",
        "evaluation_policy": "Freeze the model/augmentation choice using original train/val before scoring this holdout once; do not tune on fresh-test outcomes.",
        "model_predictions_computed": 0, "api_calls_made": 0,
        "provenance_caveat": "Fresh audit split from the same public generated/mixed archive, not newly acquired independent real-fab production data. The original splitter already accessed source labels; these selected images had no recorded prior local model evaluation in the audited artifacts.",
        "limitations": ["No lot or GAN-parent IDs are provided; D4 exclusion cannot rule out shared synthetic parentage.",
                        "Local artifact auditing cannot establish absence from model pretraining or unrecorded external evaluations.",
                        "Equal sampling per mixed combination changes the natural class distribution; normal and single-pattern cases are excluded."],
    }
    # Only after replay/exposure checks pass do we materialize new artifacts.
    for row in fresh:
        map_path = out / "maps" / f"{row['id']}.npy"
        image_path = out / "images" / f"{row['id']}.png"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(map_path, maps[row["source_index"]].astype(np.uint8), allow_pickle=False)
        row.update({"map_path": _path_label(map_path), "image_path": _path_label(image_path),
                    "holdout_role": "fresh_mixed_unscored", "parent_partition": "original_test_pool"})
    write_jsonl(out / "manifest.jsonl", fresh)
    write_json(out / "report.json", report)
    return report
