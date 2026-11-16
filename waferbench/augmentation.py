"""Training-only single-pattern composition with explicit heuristic provenance.

Structured signals below are sampled spatial hypotheses, not pixel ground truth.
The raw-union control intentionally retains compounded source background noise.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np
from scipy import ndimage

from .common import LABELS, resolve_path
from .data import dihedral_hash


def _transform(array: np.ndarray, code: int) -> np.ndarray:
    """Eight area-preserving square-grid transformations; no interpolation."""
    base = np.fliplr(array) if code >= 4 else array
    return np.ascontiguousarray(np.rot90(base, code % 4))


def _soft_structure(wafer: np.ndarray, background: float,
                    sampling_floor: float) -> np.ndarray:
    """Label-free local support, including directional support for thin traces."""
    active, failed = wafer > 0, wafer == 2
    square = np.ones((3, 3), dtype=float)
    local = ndimage.convolve(failed.astype(float), square, mode="constant")
    local /= np.maximum(1, ndimage.convolve(active.astype(float), square, mode="constant"))
    line_kernels = [np.ones((1, 5)), np.ones((5, 1)), np.eye(5), np.fliplr(np.eye(5))]
    directional = np.zeros(wafer.shape, dtype=float)
    for kernel in line_kernels:
        density = ndimage.convolve(failed.astype(float), kernel, mode="constant")
        density /= np.maximum(1, ndimage.convolve(active.astype(float), kernel, mode="constant"))
        directional = np.maximum(directional, density)
    # The observed center die contributes to each statistic. Subtract the null
    # conditional-on-failure expectation instead of treating it as extra evidence.
    local_null = background + (1 - background) / 9
    line_null = background + (1 - background) / 5
    local_evidence = np.clip((local - local_null) / max(1 - local_null, 1e-9), 0, 1)
    line_evidence = np.clip((directional - line_null) / max(1 - line_null, 1e-9), 0, 1)
    evidence = np.maximum(local_evidence, line_evidence)
    probabilities = sampling_floor + (1 - sampling_floor) * evidence
    return np.where(failed, probabilities, 0)


def _exclusive_statistics(signals: list[np.ndarray]) -> list[dict]:
    multiplicity = np.stack(signals).sum(axis=0)
    result = []
    for signal in signals:
        support = int(signal.sum())
        exclusive = int((signal & (multiplicity == 1)).sum())
        result.append({"signal_dies": support, "exclusive_signal_dies": exclusive,
                       "exclusive_signal_fraction": exclusive / support if support else 0.0})
    return result


def generate_composites(
    records: list[dict], n_per_combo: int = 100, seed: int = 20261114,
    mode: str = "structured_mix", *, background_rate: float | None = None,
    min_exclusive_fraction: float = .05, min_exclusive_dies: int = 3,
    min_wafer_fraction: float = .9, signal_sampling_floor: float = .05,
    max_attempts_per_sample: int = 20, max_outputs: int = 3999,
    max_total_attempts: int = 3999,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Compose only observed 2/3/4-pattern *training* label combinations.

    Every record must belong to train, including records used only to establish
    the supported combination vocabulary. Only single-pattern records provide
    pixels. Normal training maps estimate a shared background rate when possible.
    Zero-output and shortfall cases return empty/short arrays with an audit;
    no parent or generated image is copied to meet a requested quota.
    """
    if mode not in {"naive_union", "structured_mix"}:
        raise ValueError("mode must be naive_union or structured_mix")
    integer_settings = {"n_per_combo": n_per_combo, "min_exclusive_dies": min_exclusive_dies,
                        "max_attempts_per_sample": max_attempts_per_sample,
                        "max_outputs": max_outputs, "max_total_attempts": max_total_attempts}
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in integer_settings.values()):
        raise ValueError("Counts and attempt/output budgets must be positive integers")
    if max_outputs >= 4000 or max_total_attempts >= 4000:
        raise ValueError("A run is intentionally limited to fewer than 4000 outputs and candidate attempts")
    if not 0 <= min_exclusive_fraction <= 1 or not 0 < min_wafer_fraction <= 1:
        raise ValueError("Exclusive fraction and wafer-retention bounds must be in [0,1] and (0,1]")
    if not 0 <= signal_sampling_floor < 1:
        raise ValueError("signal_sampling_floor must be in [0,1)")
    if background_rate is not None and (not np.isfinite(background_rate) or not 0 <= background_rate < 1):
        raise ValueError("background_rate must be finite and in [0,1)")
    if not records or any(r.get("split") != "train" for r in records):
        raise ValueError("Pass train records only; validation/test parents or vocabulary are forbidden")
    order = {name: i for i, name in enumerate(LABELS)}
    ids, combos, by_label, normals = set(), set(), defaultdict(list), []
    input_group_ids, group_labels, duplicate_parent_records = set(), {}, 0
    seen_parent_groups = set()
    for row in records:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Every training record must have a unique nonempty string id")
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError(f"{row['id']}: a source group_id is required")
        labels = row.get("labels")
        if not isinstance(labels, list) or len(set(labels)) != len(labels) or set(labels) - set(LABELS):
            raise ValueError(f"{row['id']}: labels must be unique members of the canonical eight-pattern taxonomy")
        ids.add(row["id"])
        input_group_ids.add(row["group_id"])
        label_tuple = tuple(sorted(labels, key=order.__getitem__))
        if 2 <= len(labels) <= 4:
            combos.add(label_tuple)
        if len(labels) <= 1:
            previous = group_labels.setdefault(row["group_id"], label_tuple)
            if previous != label_tuple:
                raise ValueError("A parent D4 group has conflicting single-pattern labels")
            if row["group_id"] in seen_parent_groups:
                duplicate_parent_records += 1
                continue
            seen_parent_groups.add(row["group_id"])
            if labels:
                by_label[labels[0]].append(row)
            else:
                normals.append(row)
    combinations = sorted(combos, key=lambda c: (len(c), tuple(order[x] for x in c)))
    requested = len(combinations) * n_per_combo
    if requested > max_outputs:
        raise ValueError(f"Requested {requested} composites exceeds max_outputs={max_outputs}; reduce n_per_combo")
    # The vocabulary is inferred from actual train labels. Never create an
    # unsupported Random/Near-full mixture just to enumerate all possibilities.
    needed_labels = {label for combo in combinations for label in combo}
    loaded, mask_pools = {}, defaultdict(lambda: defaultdict(list))
    rejected_parents = []

    def load_parent(row):
        if row["id"] in loaded:
            return loaded[row["id"]]
        wafer = np.load(resolve_path(row["map_path"]), allow_pickle=False)
        if wafer.shape != (52, 52) or not np.isin(wafer, [0, 1, 2]).all() or not (wafer > 0).any():
            raise ValueError(f"{row['id']}: parents must be nonempty 52x52 maps with codes 0/1/2")
        wafer = wafer.astype(np.uint8, copy=True)
        loaded[row["id"]] = wafer
        return wafer

    for label in sorted(needed_labels, key=order.__getitem__):
        for row in sorted(by_label[label], key=lambda r: r["id"]):
            wafer = load_parent(row)
            if not (wafer == 2).any():
                rejected_parents.append({"id": row["id"], "label": label, "reason": "no_failing_dies"})
                continue
            mask_key = hashlib.sha256((wafer > 0).tobytes()).hexdigest()
            mask_pools[label][mask_key].append(row)
        by_label[label] = [r for pool in mask_pools[label].values() for r in pool]
    if background_rate is None:
        rates = []
        # Only normals in the supplied training set can inform this estimate.
        for row in sorted(normals, key=lambda r: r["id"]):
            wafer = load_parent(row)
            rates.append(float((wafer == 2).sum() / (wafer > 0).sum()))
        if rates:
            q_background = float(np.median(rates))
            background_source = {"method": "median_fail_density_of_training_normal_maps", "n_maps": len(rates)}
        else:
            estimates = []
            for label in sorted(needed_labels, key=order.__getitem__):
                for row in by_label[label]:
                    wafer = loaded[row["id"]]
                    active = wafer > 0
                    failed = wafer == 2
                    counts = ndimage.convolve(failed.astype(float), np.ones((3, 3)), mode="constant")
                    counts /= np.maximum(1, ndimage.convolve(active.astype(float), np.ones((3, 3)), mode="constant"))
                    estimates.append(float(np.quantile(counts[active], .2)))
            q_background = float(np.median(estimates)) if estimates else 0.0
            background_source = {"method": "median_parent_20th_percentile_local_density_no_training_normals", "n_maps": len(estimates)}
        # A background estimate at one leaves no distinguishable structure.
        q_background = float(np.clip(q_background, 0, .95))
    else:
        q_background = float(background_rate)
        background_source = {"method": "explicit_global_argument", "n_maps": 0}
    eligible_parent_ids = {r["id"] for label in needed_labels for r in by_label[label]}
    probabilities = {item_id: _soft_structure(wafer, q_background, signal_sampling_floor)
                     for item_id, wafer in loaded.items() if item_id in eligible_parent_ids}
    rng = np.random.default_rng(seed)
    # Noise draws cannot alter subsequent parent/transform draws between modes.
    background_rng = np.random.default_rng(seed ^ 0x4A31D692)
    outputs, targets, provenance = [], [], []
    generated_hashes, known_raw_groups = set(), set(input_group_ids)
    known_raw_groups.update(dihedral_hash(wafer) for wafer in loaded.values())
    reject_reasons, parent_uses, transform_uses = Counter(), Counter(), Counter()
    combo_audit, qa_examples = [], {}
    attempts = 0
    for combo in combinations:
        combo_name = "+".join(combo)
        combo_rejects, produced, local_attempts = Counter(), 0, 0
        missing = [label for label in combo if not by_label[label]]
        if missing:
            combo_audit.append({"combo": combo_name, "requested": n_per_combo, "generated": 0,
                                "shortfall": n_per_combo, "attempts": 0, "missing_parent_labels": missing})
            continue
        common_masks = sorted(set.intersection(*(set(mask_pools[label]) for label in combo)))
        weights = np.asarray([min(len(mask_pools[label][key]) for label in combo) for key in common_masks], dtype=float)
        if len(weights):
            weights /= weights.sum()
        budget = n_per_combo * max_attempts_per_sample
        while produced < n_per_combo and local_attempts < budget and attempts < max_total_attempts:
            attempts += 1
            local_attempts += 1
            if common_masks:
                chosen_mask = common_masks[int(rng.choice(len(common_masks), p=weights))]
                parents = [mask_pools[label][chosen_mask][int(rng.integers(len(mask_pools[label][chosen_mask])))] for label in combo]
            else:
                parents = [by_label[label][int(rng.integers(len(by_label[label])))] for label in combo]
            codes = rng.integers(0, 8, size=len(combo)).tolist()
            transformed = [_transform(loaded[r["id"]], code) for r, code in zip(parents, codes)]
            active_masks = [wafer > 0 for wafer in transformed]
            shared = np.logical_and.reduce(active_masks)
            shared_n = int(shared.sum())
            wafer_fractions = [shared_n / max(1, int(mask.sum())) for mask in active_masks]
            reason = None
            if not shared_n or min(wafer_fractions) < min_wafer_fraction:
                reason = "insufficient_wafer_intersection"
            # The soft selector is identical for every label, and considers only
            # local density and short-line support from the parent observation.
            scores = [_transform(probabilities[r["id"]], code) for r, code in zip(parents, codes)]
            signals = [(rng.random((52, 52)) < score) & (wafer == 2) & shared
                       for score, wafer in zip(scores, transformed)]
            exclusive = _exclusive_statistics(signals)
            if reason is None and any(not stat["signal_dies"] for stat in exclusive):
                reason = "empty_heuristic_parent_signal"
            if reason is None and any(stat["exclusive_signal_dies"] < min_exclusive_dies or
                                      stat["exclusive_signal_fraction"] < min_exclusive_fraction for stat in exclusive):
                reason = "parent_signal_occluded"
            if reason:
                combo_rejects[reason] += 1
                reject_reasons[reason] += 1
                continue
            raw_union = np.logical_or.reduce([wafer == 2 for wafer in transformed]) & shared
            signal_union = np.logical_or.reduce(signals)
            background = np.zeros((52, 52), dtype=bool)
            if mode == "naive_union":
                failed = raw_union
            else:
                background = (background_rng.random((52, 52)) < q_background) & shared & ~signal_union
                failed = signal_union | background
            composed = shared.astype(np.uint8)
            composed[failed] = 2
            output_group = dihedral_hash(composed)
            if output_group in generated_hashes or output_group in known_raw_groups:
                reason = "duplicate_composite_or_raw_training_map"
                combo_rejects[reason] += 1
                reject_reasons[reason] += 1
                continue
            generated_hashes.add(output_group)
            parent_metadata = []
            for label, row, code, wafer, score, signal, stat, wafer_fraction in zip(
                    combo, parents, codes, transformed, scores, signals, exclusive, wafer_fractions):
                original_fail = int((wafer == 2).sum())
                fail_in_shared = int(((wafer == 2) & shared).sum())
                parent_metadata.append({"label": label, "source_id": row["id"],
                    "source_group_id": row["group_id"], "source": row.get("source", "unspecified"),
                    "source_index": row.get("source_index"),
                    "source_split": "train", "source_map_path": str(row["map_path"]),
                    "d4_code": int(code), "d4_transform": f"{'flip_lr_then_' if code >= 4 else ''}rot90_{code % 4}",
                    "original_fail_dies": original_fail, "raw_fail_dies_in_shared_wafer": fail_in_shared,
                    "wafer_fraction_retained": float(wafer_fraction), **stat,
                    "sampled_signal_retention_of_shared_raw_fail": stat["signal_dies"] / max(1, fail_in_shared)})
                parent_uses[row["id"]] += 1
                transform_uses[str(code)] += 1
                if label not in qa_examples:
                    raw_qa = np.where(shared, wafer, 0).astype(np.uint8)
                    signal_qa = shared.astype(np.uint8)
                    signal_qa[signal] = 2
                    qa_examples[label] = {"label": label, "source_id": row["id"],
                        "source_group_id": row["group_id"], "d4_code": int(code),
                        "raw_map": raw_qa.tolist(), "sampled_signal_map": signal_qa.tolist(),
                        "keep_probability": np.round(score, 4).tolist(),
                        "interpretation": "Human QA only: sampled spatial evidence, not a labeled defect segmentation."}
            item_id = f"synthetic-{mode}-{len(outputs):05d}"
            signature = hashlib.sha256("|".join(f"{r['id']}:{code}" for r, code in zip(parents, codes)).encode()).hexdigest()
            provenance.append({"id": item_id, "source": "synthetic_training_composite", "split": "train",
                "mode": mode, "combo": combo_name, "labels": list(combo), "group_id": output_group,
                "candidate_sequence_number": attempts, "parent_transform_signature": signature,
                "parents": parent_metadata, "background_rate": q_background,
                "wafer_active_dies": shared_n, "raw_failed_union_dies": int(raw_union.sum()),
                "heuristic_signal_union_dies": int(signal_union.sum()),
                "new_shared_background_dies": int(background.sum()), "output_fail_dies": int(failed.sum()),
                "raw_failed_union_density": float(raw_union.sum() / shared_n),
                "output_fail_density": float(failed.sum() / shared_n),
                "heuristic_signal_exclusive_support_accepted": True})
            outputs.append(composed)
            targets.append([int(label in combo) for label in LABELS])
            produced += 1
        combo_audit.append({"combo": combo_name, "requested": n_per_combo, "generated": produced,
                            "shortfall": n_per_combo - produced, "attempts": local_attempts,
                            "rejection_counts": dict(combo_rejects), "same_raw_wafer_template_available": bool(common_masks)})
    maps_array = np.stack(outputs).astype(np.uint8) if outputs else np.empty((0, 52, 52), dtype=np.uint8)
    targets_array = np.asarray(targets, dtype=np.uint8).reshape(-1, len(LABELS))
    parent_statistics = defaultdict(list)
    for row in provenance:
        for parent in row["parents"]:
            parent_statistics[parent["label"]].append(parent)
    audit = {"mode": mode, "seed": seed, "label_order": list(LABELS), "input_records": len(records),
        "input_split_counts": {"train": len(records)}, "requested_total": requested,
        "generated_total": len(outputs), "total_shortfall": requested - len(outputs),
        "candidate_attempts": attempts, "max_total_attempts": max_total_attempts,
        "attempt_budget_exhausted": attempts >= max_total_attempts and len(outputs) < requested,
        "rejection_counts": dict(reject_reasons), "combinations": combo_audit,
        "background_rate": q_background, "background_estimate": background_source,
        "parameters": {"n_per_combo": n_per_combo, "min_exclusive_fraction": min_exclusive_fraction,
            "min_exclusive_dies": min_exclusive_dies, "min_wafer_fraction": min_wafer_fraction,
            "signal_sampling_floor": signal_sampling_floor, "max_attempts_per_sample": max_attempts_per_sample,
            "max_outputs": max_outputs},
        "unique_single_parent_counts": {label: len(by_label[label]) for label in sorted(needed_labels, key=order.__getitem__)},
        "duplicate_parent_group_records_removed": duplicate_parent_records,
        "rejected_parent_records": rejected_parents,
        "parent_id_usage": dict(sorted(parent_uses.items())), "d4_transform_counts": dict(sorted(transform_uses.items())),
        "by_parent_label": {label: {"n_parent_uses": len(items),
            "mean_signal_retention_of_shared_raw_fail": float(np.mean([r["sampled_signal_retention_of_shared_raw_fail"] for r in items])),
            "minimum_exclusive_signal_fraction": min(r["exclusive_signal_fraction"] for r in items),
            "mean_exclusive_signal_fraction": float(np.mean([r["exclusive_signal_fraction"] for r in items])),
            "minimum_wafer_fraction_retained": min(r["wafer_fraction_retained"] for r in items)}
            for label, items in sorted(parent_statistics.items())},
        "signal_qa_examples": [qa_examples[label] for label in LABELS if label in qa_examples],
        "invariants": {"parents_train_only": True, "vocabulary_from_train_only": True,
            "unique_generated_D4_groups": len(generated_hashes), "duplicates_padded_to_quota": 0,
            "parent_geometry": "52x52, D4 only, shared active-wafer intersection; no interpolation or label-changing translation"},
        "heuristic_assumptions": [
            "Single-pattern parent labels are assumed correct; pixel-level signal/background labels are unavailable.",
            "A label-free soft selector uses excess 3x3 density and strongest five-die directional density above a background null; it can retain noise or miss true pattern dies.",
            "Thin coherent traces are supported by directional evidence rather than removed by a Potts smoothness penalty; this does not prove preservation of every scratch.",
            "Exclusive sampled support is an occlusion proxy, not proof that a classifier can identify every mixed component.",
            "Structured mode adds independent shared background once after union of sampled parent signals; residual parent noise may remain in those signals.",
            "Naive-union mode deliberately unions original failing dies, including each parent's noise, and adds no separate background.",
            "D4 transforms preserve center/ring/interior/edge locations; an off-center Loc is never translated toward an Edge-Loc or vice versa.",
            "Outputs remain dependent on their training parents and are not independent real-fab examples or a held-out test set.",
        ]}
    return maps_array, targets_array, provenance, audit
