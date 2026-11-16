"""Run an entirely local smoke demo without downloading wafer datasets.

From a fresh clone with the project's dependencies installed:

    python examples/smoke_demo.py --out smoke-output

The procedural maps only exercise the pipeline. Their scores are not research
results, and the synthetic labels are not expert production annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

# Make direct script invocation work from an uninstalled fresh source checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _toy_map(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.indices((52, 52), dtype=float)
    x, y = (xx-25.5)/25.5, (yy-25.5)/25.5
    radius, angle = np.hypot(x, y), np.arctan2(y, x)
    active = radius <= .95
    masks = {
        "center": radius < .23,
        "donut": (radius > .40) & (radius < .53),
        "edge_loc": (radius > .73) & (angle > -.3) & (angle < .35),
        "edge_ring": radius > .83,
        "loc": ((x+.42)**2 + (y-.21)**2) < .16**2,
        "near_full": rng.random(active.shape) < .94,
        "scratch": (np.abs(y-.45*x-.12) < .035) & (np.abs(x) < .85),
        "random": rng.random(active.shape) < .23,
    }
    failed = np.zeros(active.shape, dtype=bool)
    for label in labels:
        failed |= masks[label]
    # Independent small background samples make every source map/D4 group unique.
    indices = np.flatnonzero(active)
    failed.flat[rng.choice(indices, size=7, replace=False)] = True
    wafer = active.astype(np.uint8)
    wafer[failed & active] = 2
    return wafer


def make_toy_fixture(out: Path, seed: int = 721) -> Path:
    """Write 40 local maps: nine base classes plus one mixed combination.

    Every class/combination has train=2, val=1, test=1. All map paths are absolute
    so commands can run from an unrelated empty working directory. D4 hashes are
    computed from the maps themselves and are unique across every split.
    """
    from waferbench.common import LABELS, SINGLE_LABELS, write_jsonl
    from waferbench.data import dihedral_hash

    out = Path(out).resolve()
    manifest = out / "manifest.jsonl"
    if manifest.exists():
        raise FileExistsError("Demo manifest already exists; choose a fresh --out directory")
    rng = np.random.default_rng(seed)
    classes = [([] if category == "normal" else [category]) for category in SINGLE_LABELS]
    classes.append(["center", "scratch"])
    rows, seen = [], set()
    for class_index, labels in enumerate(classes):
        for split, count in [("train", 2), ("val", 1), ("test", 1)]:
            for sample in range(count):
                for _ in range(100):
                    wafer = _toy_map(labels, rng)
                    group = dihedral_hash(wafer)
                    if group not in seen:
                        break
                else:
                    raise RuntimeError("Could not generate a distinct procedural D4 group")
                seen.add(group)
                identifier = f"demo-{class_index:02d}-{split}-{sample}"
                path = out / "maps" / f"{identifier}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, wafer, allow_pickle=False)
                rows.append({"id": identifier, "source": "procedural_smoke_demo_only",
                             "split": split, "labels": labels.copy(),
                             "combo": "+".join(labels) or "normal", "group_id": group,
                             "map_path": str(path),
                             "image_path": str(out / "images" / f"{identifier}.png")})
    assert all(set(row["labels"]).issubset(LABELS) for row in rows)
    write_jsonl(manifest, rows)
    return manifest


def run_demo(out: Path) -> dict:
    from waferbench.advanced_features import advanced_features
    from waferbench.baseline import run_baseline
    from waferbench.common import read_jsonl

    out = Path(out).resolve()
    manifest = make_toy_fixture(out / "data")
    rows = read_jsonl(manifest)
    results = {}
    for name, task, features in [("single", "single", "base"), ("multi", "multi", "base"),
                                 ("multi_advanced", "multi", "advanced")]:
        metrics = run_baseline(manifest, out / "runs" / name, task, seed=721, features=features)
        results[name] = {"test_samples": metrics["n"], "feature_set": features,
                         "probability_columns": len(metrics["labels"]),
                         "feature_count": metrics["run"]["n_features"]}
    first = np.load(rows[0]["map_path"], allow_pickle=False)
    return {"purpose": "Procedural pipeline smoke test only; these scores are not wafer benchmark results.",
            "output_directory": str(out), "manifest": str(manifest),
            "maps": len(rows), "split_sizes": dict(Counter(row["split"] for row in rows)),
            "advanced_feature_count": int(advanced_features(first).size),
            "api_calls_made": 0, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New output directory; defaults to a fresh temporary directory")
    args = parser.parse_args()
    out = args.out if args.out is not None else Path(tempfile.mkdtemp(prefix="wafer-rf-smoke-"))
    print(json.dumps(run_demo(out), indent=2))


if __name__ == "__main__":
    main()
