"""WM-811K ingestion with native-test preservation and lot-disjoint evaluation."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import LABELS, ROOT, resolve_path, write_json, write_jsonl
from .data import make_contact_sheet, render_map, sha256_file


def _read_legacy_frame(source: Path):
    """Read the original pandas pickle without installing process-wide aliases."""
    import pandas as pd
    from pandas.compat.pickle_compat import Unpickler

    class WM811KUnpickler(Unpickler):
        def find_class(self, module, name):
            # Original WM-811K predates pandas' pandas.core.indexes namespace.
            if module.startswith("pandas.indexes."):
                module = module.replace("pandas.indexes.", "pandas.core.indexes.", 1)
            return super().find_class(module, name)

    try:
        return pd.read_pickle(source)
    except (ImportError, ModuleNotFoundError, AttributeError, UnicodeDecodeError):
        with source.open("rb") as handle:
            return WM811KUnpickler(handle, encoding="latin1").load()


def _text(value) -> str:
    """Flatten empty/singleton numpy and MATLAB-style text containers."""
    if isinstance(value, np.ndarray):
        parts = [_text(v) for v in value.ravel()]
    elif isinstance(value, (list, tuple)):
        parts = [_text(v) for v in value]
    elif value is None:
        return ""
    elif isinstance(value, bytes):
        return value.decode("utf-8").strip()
    else:
        return str(value).strip()
    parts = [p for p in parts if p]
    if len(set(parts)) > 1:
        raise ValueError(f"Expected one text label, got {parts!r}")
    return parts[0] if parts else ""


def _d4_hash(wafer: np.ndarray) -> str:
    """Canonicalize each transform's dimensions, including rectangular maps."""
    versions = []
    for base in (wafer, wafer.T):
        for turns in range(4):
            v = np.ascontiguousarray(np.rot90(base, turns), dtype=np.uint8)
            versions.append(f"{v.shape[0]},{v.shape[1]}:".encode() + v.tobytes())
    return hashlib.sha256(min(versions)).hexdigest()


def _validation_lots(lot_hist: dict[str, np.ndarray], seed: int,
                     train_cap: int, val_cap: int) -> tuple[set[str], dict]:
    """Search deterministic *lot* assignments before taking any wafer samples."""
    lots = sorted(lot_hist)
    if len(lots) < 2:
        raise ValueError("Need at least two non-test production lots for train/validation.")
    hist = np.stack([lot_hist[lot] for lot in lots])
    totals = hist.sum(axis=0)
    n_val = max(1, min(len(lots) - 1, round(len(lots) * .15)))
    train_goal = np.minimum(train_cap if train_cap else totals, np.maximum(1, np.floor(totals * .85)))
    val_goal = np.minimum(val_cap if val_cap else totals, np.maximum(1, np.ceil(totals * .15)))
    rng = np.random.default_rng(seed)
    best = None
    # These are candidate group assignments, not model-based selection or test tuning.
    for _ in range(512):
        chosen = rng.permutation(len(lots))[:n_val]
        vc = hist[chosen].sum(axis=0)
        tc = totals - vc
        coverage = (np.minimum(vc / np.maximum(val_goal, 1), 1).sum()
                    + np.minimum(tc / np.maximum(train_goal, 1), 1).sum())
        distance = np.abs(vc / np.maximum(totals, 1) - .15).sum()
        score = float(coverage - .001 * distance)
        if best is None or score > best[0]:
            best = (score, chosen, vc, tc)
    _, chosen, vc, tc = best
    return {lots[i] for i in chosen}, {
        "algorithm": "512 seeded 15%-of-lots candidate assignments; maximize class coverage in train/validation, then balance class proportions",
        "candidate_assignments": 512,
        "class_order": ["normal", *LABELS],
        "validation_target": val_goal.astype(int).tolist(),
        "available_before_d4_quarantine": {"train": tc.tolist(), "val": vc.tolist()},
    }


def prepare_wm811k(source: Path, out: Path, seed: int = 42,
                   train_per_class: int = 200, val_per_class: int = 20,
                   test_per_class: int = 20) -> dict:
    """Prepare real wafer maps; lots and D4-equivalent maps never cross splits."""
    source, out = resolve_path(source), resolve_path(out)
    if (out / "manifest.jsonl").exists():
        raise FileExistsError(f"{out / 'manifest.jsonl'} already exists; choose a new output directory.")
    # Every emitted path must be usable by the shared root-relative manifest reader.
    out.relative_to(ROOT)
    caps = {"train": train_per_class, "val": val_per_class, "test": test_per_class}
    if any(not isinstance(cap, int) or cap < 0 for cap in caps.values()):
        raise ValueError("Per-class caps must be non-negative integers; zero means uncapped.")
    frame = _read_legacy_frame(source)
    required = {"waferMap", "failureType", "lotName", "waferIndex", "trianTestLabel"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing WM-811K columns: {sorted(required - set(frame.columns))}")
    native_stats, raw_labels, labeled_native = Counter(), Counter(), Counter()
    records, invalid = [], []
    shape_counts, pixel_counts = Counter(), Counter()
    lookup = {name.replace("_", "-"): name for name in LABELS}
    lookup["none"] = "normal"
    # Use positional indices because those identify rows in the original pickle.
    for idx, (wafer, label, lot, wafer_index, official) in enumerate(zip(
            frame.waferMap, frame.failureType, frame.lotName,
            frame.waferIndex, frame.trianTestLabel)):
        label, official, lot = _text(label), _text(official), _text(lot)
        native_stats[official or "unassigned"] += 1
        raw_labels[label or "unlabeled"] += 1
        if not label:
            continue
        labeled_native[official or "unassigned"] += 1
        key = label.lower().replace("_", "-")
        if key not in lookup:
            raise ValueError(f"Unrecognized failureType {label!r} at row {idx}.")
        if not lot:
            raise ValueError(f"Labeled wafer row {idx} has no lot identifier.")
        wafer = np.asarray(wafer)
        if wafer.ndim != 2 or not wafer.size:
            invalid.append({"source_index": idx, "reason": "not a nonempty 2-D map"})
            continue
        vals, counts = np.unique(wafer, return_counts=True)
        pixel_counts.update({str(v): int(n) for v, n in zip(vals.tolist(), counts.tolist())})
        shape_counts[f"{wafer.shape[0]}x{wafer.shape[1]}"] += 1
        if not np.isin(vals, [0, 1, 2]).all():
            invalid.append({"source_index": idx, "reason": "unexpected die code", "values": vals.tolist()})
            continue
        combo = lookup[key]
        records.append({"source_index": idx, "lot_name": lot,
                        "wafer_index": float(wafer_index), "official_split": official,
                        "combo": combo, "labels": [] if combo == "normal" else [combo]})
    labeled_count = len(frame) - raw_labels["unlabeled"]
    raw_native_lots = defaultdict(set)
    for record in records:
        raw_native_lots[record["official_split"] or "unassigned"].add(record["lot_name"])
    training_lots = raw_native_lots.get("Training", set())
    native_test_lots = raw_native_lots.get("Test", set())
    if not native_test_lots:
        raise ValueError("No native Test lots found; refusing to invent an undocumented test partition.")
    overlapping_lots = training_lots & native_test_lots
    all_lots = {r["lot_name"] for r in records}
    train_val_lots = all_lots - native_test_lots
    classes = ["normal", *LABELS]
    class_index = {label: i for i, label in enumerate(classes)}
    lot_hist = {lot: np.zeros(len(classes), dtype=np.int64) for lot in train_val_lots}
    for record in records:
        if record["lot_name"] in lot_hist:
            lot_hist[record["lot_name"]][class_index[record["combo"]]] += 1
    val_lots, assignment_report = _validation_lots(lot_hist, seed, train_per_class, val_per_class)
    groups = defaultdict(list)
    for record in records:
        lot = record["lot_name"]
        record["split"] = "test" if lot in native_test_lots else ("val" if lot in val_lots else "train")
        wafer = np.asarray(frame.waferMap.iloc[record["source_index"]], dtype=np.uint8)
        record["group_id"] = _d4_hash(wafer)
        groups[record["group_id"]].append(record)
    quarantined_groups, representatives = [], []
    quarantine_reasons = Counter()
    for group_id, members in groups.items():
        reasons = []
        if len({m["combo"] for m in members}) > 1:
            reasons.append("conflicting_labels")
        if len({m["split"] for m in members}) > 1:
            reasons.append("cross_split_d4_duplicate")
        if reasons:
            quarantine_reasons.update(reasons)
            quarantined_groups.append({"group_id": group_id, "reasons": reasons,
                                       "source_indices": [m["source_index"] for m in members],
                                       "splits": sorted({m["split"] for m in members})})
            continue
        representative = min(members, key=lambda m: m["source_index"]).copy()
        representative["duplicate_multiplicity"] = len(members)
        representatives.append(representative)
    rng = np.random.default_rng(seed)
    rows, inventory = [], []
    for combo in classes:
        counts = {}
        for split in ("train", "val", "test"):
            pool = sorted((r for r in representatives if r["combo"] == combo and r["split"] == split),
                          key=lambda r: r["group_id"])
            rng.shuffle(pool)
            selected = pool[:caps[split]] if caps[split] else pool
            counts[split] = {"available": len(pool), "selected": len(selected),
                             "requested": caps[split], "shortfall": max(0, caps[split] - len(selected))}
            for record in selected:
                idx = record["source_index"]
                item_id = f"wm811k-{idx:06d}"
                map_path, image_path = out / "maps" / f"{item_id}.npy", out / "images" / f"{item_id}.png"
                map_path.parent.mkdir(parents=True, exist_ok=True)
                wafer = np.asarray(frame.waferMap.iloc[idx], dtype=np.uint8)
                np.save(map_path, wafer, allow_pickle=False)
                render_map(wafer, image_path)
                rows.append({"id": item_id, "source": "WM811K", **record,
                             "map_path": map_path.relative_to(ROOT).as_posix(),
                             "image_path": image_path.relative_to(ROOT).as_posix()})
        inventory.append({"combo": combo, "splits": counts})
    rows.sort(key=lambda r: r["id"])
    pilot = []
    for combo in classes:
        pilot.extend(sorted((r for r in rows if r["combo"] == combo and r["split"] == "test"),
                            key=lambda r: r["group_id"])[:5])
    split_lots = {s: {r["lot_name"] for r in rows if r["split"] == s} for s in caps}
    split_groups = {s: {r["group_id"] for r in rows if r["split"] == s} for s in caps}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    lot_overlap = {f"{a}_{b}": len(split_lots[a] & split_lots[b]) for a, b in pairs}
    d4_overlap = {f"{a}_{b}": len(split_groups[a] & split_groups[b]) for a, b in pairs}
    if any(lot_overlap.values()) or any(d4_overlap.values()):
        raise AssertionError("Leakage detected in prepared WM-811K manifests.")
    moved = Counter((r["official_split"] or "unassigned") + "->" + r["split"] for r in records)
    report = {
        "dataset": "WM811K", "source_file": str(source), "source_sha256": sha256_file(source),
        "raw_rows": len(frame), "raw_columns": list(frame.columns),
        "raw_labeled_rows": labeled_count, "raw_unlabeled_rows": raw_labels["unlabeled"],
        "raw_failure_type_counts": dict(sorted(raw_labels.items())),
        "raw_official_split_counts": dict(sorted(native_stats.items())),
        "labeled_official_split_counts": dict(sorted(labeled_native.items())),
        "labeled_pixel_value_counts": dict(sorted(pixel_counts.items())),
        "labeled_map_shape_counts": dict(sorted(shape_counts.items())),
        "quarantined_invalid_code_or_shape_maps": len(invalid),
        "native_labeled_lot_counts": {k: len(v) for k, v in sorted(raw_native_lots.items())},
        "native_training_test_overlap_lots": len(overlapping_lots),
        "native_training_test_overlap_lot_names": sorted(overlapping_lots),
        "official_to_group_partition_counts_before_d4_quarantine": dict(sorted(moved.items())),
        "lot_partition_counts_before_d4_quarantine": {"train": len(train_val_lots - val_lots), "val": len(val_lots), "test": len(native_test_lots)},
        "validation_lot_assignment": assignment_report,
        "d4_unique_groups_before_quarantine": len(groups),
        "quarantined_d4_groups": len(quarantined_groups),
        "quarantined_d4_reason_counts": dict(quarantine_reasons),
        "quarantined_d4_maps": sum(len(g["source_indices"]) for g in quarantined_groups),
        "selected_split_counts": dict(Counter(r["split"] for r in rows)),
        "selected_lot_counts": {s: len(v) for s, v in split_lots.items()},
        "pilot_samples": len(pilot), "seed": seed, "caps_per_class": caps, "inventory": inventory,
        "leakage_checks": {"lot_overlap": lot_overlap, "exact_rotated_mirrored_overlap": d4_overlap,
            "policy": "Reserve every lot containing a native Test record for test; split other lots into train/validation before capping. Quarantine every D4 group spanning splits or labels; keep one representative from each remaining D4 group.",
            "limitation": "Lot/D4 isolation cannot establish absence from a foundation model's pretraining data; explicit none labels mean no assigned systematic pattern, not zero failing dies."},
        "provenance_caveat": "Real-world fab maps from WM-811K. This is a capped lot-disjoint benchmark derived from the native partition, not the complete native test benchmark.",
    }
    write_jsonl(out / "manifest.jsonl", rows)
    write_jsonl(out / "pilot.jsonl", pilot)
    write_json(out / "data_report.json", report)
    write_json(out / "quarantined_invalid_maps.json", invalid)
    write_json(out / "quarantined_d4_groups.json", quarantined_groups)
    make_contact_sheet(pilot, out / "preview.png")
    return report
