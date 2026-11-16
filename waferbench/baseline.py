"""A lightweight supervised spatial-feature/random-forest comparison baseline."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .metrics import evaluate_predictions

PATTERNS = ["center", "donut", "edge_loc", "edge_ring", "loc", "near_full", "scratch", "random"]
FEATURE_VERSION = "spatial_v1"


def spatial_features(wafer_map: np.ndarray) -> np.ndarray:
    """Deterministic spatial statistics, radial/angular profiles and coarse grids."""
    wafer_map = np.asarray(wafer_map).squeeze()
    if wafer_map.ndim != 2 or not np.isin(wafer_map, [0, 1, 2]).all():
        raise ValueError("A wafer map must be 2D with 0=outside, 1=good, 2=defective")
    active, defect = wafer_map > 0, wafer_map == 2
    if not active.any():
        raise ValueError("A wafer map contains no active dies")
    yy, xx = np.indices(wafer_map.shape)
    rows, cols = np.nonzero(active)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    ry, rx = max((rows.max() - rows.min()) / 2, 1), max((cols.max() - cols.min()) / 2, 1)
    y, x = (yy - cy) / ry, (xx - cx) / rx
    radius, angle = np.sqrt(x*x + y*y), (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
    count, total = float(defect.sum()), float(active.sum())
    result = [count/total, total/wafer_map.size, ry/rx]

    def densities(regions):
        return [float((defect & mask).sum() / max(int((active & mask).sum()), 1)) for mask in regions]

    # Fractions of defective dies in each region, normalized by available dies.
    result.extend(densities([(radius >= a) & (radius < b) for a, b in zip(np.linspace(0, 1.5, 13)[:-1], np.linspace(0, 1.5, 13)[1:])]))
    result.extend(densities([(angle >= a) & (angle < b) for a, b in zip(np.linspace(0, 2*np.pi, 17)[:-1], np.linspace(0, 2*np.pi, 17)[1:])]))
    for n in [4, 8]:
        edges = np.linspace(-1.001, 1.001, n+1)
        result.extend(densities([(y >= ya) & (y < yb) & (x >= xa) & (x < xb)
                                 for ya, yb in zip(edges[:-1], edges[1:])
                                 for xa, xb in zip(edges[:-1], edges[1:])]))
    edges = np.linspace(-1.001, 1.001, 17)
    result.extend(densities([(y >= a) & (y < b) for a,b in zip(edges[:-1], edges[1:])]))
    result.extend(densities([(x >= a) & (x < b) for a,b in zip(edges[:-1], edges[1:])]))
    if count:
        dx, dy, dr = x[defect], y[defect], radius[defect]
        covariance = np.cov(np.stack([dx, dy]), bias=True) if count > 1 else np.zeros((2,2))
        eigen = np.linalg.eigvalsh(covariance)
        result.extend([dx.mean(), dy.mean(), dx.std(), dy.std(), covariance[0,1],
                       eigen[0], eigen[1], eigen[0]/max(eigen[1], 1e-9),
                       dr.mean(), dr.std(), *np.quantile(dr, [0.1,0.25,0.5,0.75,0.9])])
    else:
        result.extend([0.0]*15)
    # Small isolated failures versus contiguous cluster/ring/scratch structures.
    for iterations in [0, 1]:
        connected = defect if not iterations else ndimage.binary_dilation(defect, iterations=iterations) & active
        components, component_count = ndimage.label(connected, structure=np.ones((3,3)))
        sizes = np.bincount(components.ravel())[1:]
        if component_count:
            sizes = np.sort(sizes)[::-1]
            result.extend([component_count/total, sizes[0]/total, float(np.median(sizes))/total,
                           sizes[:3].sum()/total, float((sizes == 1).sum())/component_count])
        else:
            result.extend([0.0]*5)
    return np.nan_to_num(np.asarray(result, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _map_path(row: dict, manifest: Path) -> Path:
    path = Path(row["map_path"])
    candidates = [path] if path.is_absolute() else [Path.cwd()/path, manifest.parent/path, *(parent/path for parent in manifest.parents)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Map for {row['id']} not found: {path}")


def _dump_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _dump_rows(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _thresholds(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    chosen = np.full(y.shape[1], .5)
    grid = np.linspace(.1, .9, 33)
    for i in range(y.shape[1]):
        # Without both positive and negative validation cases, retain preregistered 0.5.
        if len(np.unique(y[:, i])) != 2:
            continue
        scored = [(f1_score(y[:, i], probabilities[:, i] >= threshold, zero_division=0),
                   -abs(threshold-.5), threshold) for threshold in grid]
        chosen[i] = max(scored)[2]
    return chosen


def _predict_probabilities(model, features: np.ndarray, task: str, classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(features)
    if task == "single":
        output = np.zeros((len(features), len(classes)))
        for j, label in enumerate(model.classes_):
            output[:, classes.index(str(label))] = raw[:, j]
    else:
        output = np.zeros((len(features), len(classes)))
        for j, (values, options) in enumerate(zip(raw, model.classes_)):
            if 1 in options:
                output[:, j] = values[:, list(options).index(1)]
    return output


def run_baseline(manifest: Path, output_dir: Path, task: str, seed: int = 42, features: str = "base") -> dict:
    """Fit using train only; tune multi-label thresholds on val; score held-out test.

    Saves a resumable prediction file and enough metadata to reproduce the run.
    This is a supervised handcrafted-feature baseline, not a zero-shot model.
    """
    if task not in {"single", "multi"}:
        raise ValueError("task must be single or multi")
    if features not in {"base", "advanced"}:
        raise ValueError("features must be base or advanced")
    if features == "advanced":
        from .advanced_features import advanced_features, FEATURE_VERSION as selected_feature_version
        extractor = advanced_features
    else:
        extractor, selected_feature_version = spatial_features, FEATURE_VERSION
    manifest, output_dir = Path(manifest).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    all_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows = [row for row in all_rows if task != "single" or len(row["labels"]) <= 1]
    if any(set(row["labels"]) - set(PATTERNS) for row in rows):
        raise ValueError("Manifest contains labels outside the canonical 8-pattern taxonomy")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ids in manifest")
    classes = ["normal", *PATTERNS] if task == "single" else PATTERNS.copy()
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ["train", "val", "test"]}
    if not all(split_rows.values()):
        raise ValueError("Manifest requires nonempty train, val, and test splits")
    groups = {split: {row["group_id"] for row in selected} for split, selected in split_rows.items()}
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError("Group leakage across train/val/test splits")
    # Manifest fingerprint includes all map references and split/label assignments.
    # Prepared maps are immutable; re-prepare the manifest after changing source data.
    cache_key = hashlib.sha256((selected_feature_version + manifest.read_text(encoding="utf-8-sig") + task).encode()).hexdigest()[:16]
    cache = output_dir / f"features_{cache_key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as existing:
            features = existing["features"]
        if features.shape[0] != len(rows):
            raise ValueError("Feature cache row count mismatch")
    else:
        features = np.stack([extractor(np.load(_map_path(row, manifest), allow_pickle=False)) for row in rows])
        np.savez_compressed(cache, features=features)
    split_indices = {split: np.asarray([i for i, row in enumerate(rows) if row["split"] == split]) for split in split_rows}
    if task == "single":
        target = np.asarray([row["labels"][0] if row["labels"] else "normal" for row in rows])
        missing_train = set(classes) - set(target[split_indices["train"]])
        if missing_train:
            raise ValueError(f"Classes absent from training split: {missing_train}")
    else:
        target = np.asarray([[int(label in row["labels"]) for label in classes] for row in rows])
    parameters = {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt",
                  "class_weight": "balanced_subsample" if task == "single" else None,
                  "random_state": seed, "n_jobs": -1}
    model = RandomForestClassifier(**parameters)
    training_start = time.perf_counter()
    model.fit(features[split_indices["train"]], target[split_indices["train"]])
    training_seconds = time.perf_counter() - training_start
    probabilities = {split: _predict_probabilities(model, features[indices], task, classes)
                     for split, indices in split_indices.items() if split != "train"}
    thresholds = _thresholds(target[split_indices["val"]], probabilities["val"]) if task == "multi" else None

    def prediction_rows(split, threshold=None):
        result = []
        for row, probs in zip(split_rows[split], probabilities[split]):
            if task == "single":
                prediction = classes[int(probs.argmax())]
                predicted_labels = [] if prediction == "normal" else [prediction]
            else:
                predicted_labels = [label for label, p, cutoff in zip(classes, probs, threshold) if p >= cutoff]
            result.append({"id": row["id"], "labels": row["labels"], "predicted_labels": predicted_labels,
                           "probabilities": {label: float(p) for label, p in zip(classes, probs)},
                           "split": split, "task": task, "backend": "spatial_random_forest", "status": "ok"})
        return result

    test_records = prediction_rows("test", thresholds)
    val_records = prediction_rows("val", thresholds)
    metrics = evaluate_predictions(test_records, classes, task)
    feature_set = "base" if selected_feature_version == FEATURE_VERSION else "advanced"
    metrics["run"] = {"backend": "spatial_random_forest", "feature_version": selected_feature_version,
                      "feature_set": feature_set,
                      "n_features": int(features.shape[1]), "seed": seed, "manifest": str(manifest),
                      "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "split_sizes": {split: len(selected) for split, selected in split_rows.items()},
                      "parameters": parameters, "train_seconds": training_seconds,
                      "elapsed_seconds": time.perf_counter() - start,
                      "selection": "single task excludes wafers with >=2 labels; multi includes normal/single/mixed"}
    if thresholds is not None:
        metrics["thresholds"] = {label: float(value) for label, value in zip(classes, thresholds)}
        metrics["threshold_selection"] = "independent per-label validation F1; grid 0.1..0.9 step0.025; ties nearest0.5"
        fixed = prediction_rows("test", np.full(len(classes), .5))
        metrics["fixed_0_5"] = evaluate_predictions(fixed, classes, task)
        _dump_rows(output_dir/"predictions_fixed_0_5.jsonl", fixed)
    _dump_rows(output_dir/"predictions.jsonl", test_records)
    _dump_rows(output_dir/"val_predictions.jsonl", val_records)
    _dump_json(output_dir/"metrics.json", metrics)
    import joblib
    metadata = {"task": task, "classes": classes, "feature_set": feature_set,
                "feature_version": selected_feature_version, "n_features": int(features.shape[1]),
                "thresholds": None if thresholds is None else thresholds.tolist(),
                "seed": seed, "parameters": parameters,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    joblib.dump({"model": model, **metadata}, output_dir/"model.joblib", compress=3)
    _dump_json(output_dir/"model_metadata.json", metadata)
    return metrics
