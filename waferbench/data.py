"""Verified MixedWM38 ingestion, duplicate grouping, and label-free image rendering."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import LABELS, ROOT, resolve_path, write_json, write_jsonl


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dihedral_hash(wafer: np.ndarray) -> str:
    """Keep identical, rotated and mirrored wafers in the same split."""
    versions = [np.rot90(wafer,k) for k in range(4)]
    versions += [np.rot90(wafer.T,k) for k in range(4)]
    # Include each transformed shape, so a rectangular 90-degree rotation has
    # the same canonical representation without conflating reshaped arrays.
    encodings = [str(v.shape).encode()+b":"+np.ascontiguousarray(v).tobytes() for v in versions]
    return hashlib.sha256(min(encodings)).hexdigest()


def render_map(wafer: np.ndarray, path: Path, palette: str = "cyan", size: int = 416) -> None:
    """Nearest-neighbor only; no axes, captions, filenames or labels in model input."""
    colors = [(255,255,255),(0,224,229),(188,0,239)] if palette == "cyan" else [(255,255,255),(216,216,216),(30,30,30)]
    rgb = np.asarray(colors, dtype=np.uint8)[np.asarray(wafer, dtype=np.uint8)]
    im = Image.fromarray(rgb).resize((size,size), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def prepare_mixed(source: Path, out: Path, seed: int = 42,
                  train_per_combo: int = 200, val_per_combo: int = 20,
                  test_per_combo: int = 20) -> dict:
    source, out = resolve_path(source), resolve_path(out)
    out.relative_to(ROOT)
    if any(not isinstance(cap,int) or cap<0 for cap in [train_per_combo,val_per_combo,test_per_combo]):
        raise ValueError("Per-combination caps must be non-negative integers; zero means uncapped")
    manifest = out / "manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"{manifest} already exists; choose a new --out directory to preserve the split.")
    archive = np.load(source, allow_pickle=False)
    maps, targets = archive["arr_0"], archive["arr_1"]
    if maps.ndim != 3 or targets.shape != (len(maps),len(LABELS)):
        raise ValueError(f"Unexpected MixedWM38 shapes: {maps.shape}, {targets.shape}")
    if not np.isin(targets, [0,1]).all():
        raise ValueError("Targets must be eight binary components in the verified author order.")
    # Unexpected die codes are quarantined rather than silently reinterpreted.
    valid = np.isin(maps,[0,1,2]).all(axis=(1,2))
    groups = defaultdict(list)
    for idx in np.flatnonzero(valid):
        groups[dihedral_hash(maps[idx].astype(np.uint8))].append(int(idx))
    conflicts, unique = [], defaultdict(list)
    for group, indices in groups.items():
        label_sets = {tuple(int(v) for v in targets[i]) for i in indices}
        if len(label_sets) > 1:
            conflicts.append({"group_id":group,"indices":indices})
            continue
        unique[next(iter(label_sets))].append((group, indices[0], len(indices)))
    rng = np.random.default_rng(seed)
    rows, inventory = [], []
    for bits, members in sorted(unique.items()):
        members.sort()
        rng.shuffle(members)
        n = len(members)
        if n < 7:
            raise ValueError(f"Insufficient unique maps for a three-way split: {bits}: {n}")
        n_test, n_val = max(1,round(n*.15)), max(1,round(n*.15))
        partitions = {"test":members[:n_test],"val":members[n_test:n_test+n_val],"train":members[n_test+n_val:]}
        labels = [name for name,bit in zip(LABELS,bits) if bit]
        combo = "+".join(labels) or "normal"
        counts = {}
        for split, pool in partitions.items():
            cap = {"train":train_per_combo,"val":val_per_combo,"test":test_per_combo}[split]
            selected = pool[:cap] if cap else pool
            counts[split] = {"available":len(pool),"selected":len(selected)}
            for group, idx, multiplicity in selected:
                item_id = f"mixedwm38-{idx:05d}"
                map_path, image_path = out/"maps"/f"{item_id}.npy", out/"images"/f"{item_id}.png"
                map_path.parent.mkdir(parents=True,exist_ok=True)
                np.save(map_path,maps[idx].astype(np.uint8),allow_pickle=False)
                if split != "train":
                    render_map(maps[idx],image_path)
                rows.append({"id":item_id,"source":"MixedWM38","source_index":idx,"map_path":map_path.relative_to(ROOT).as_posix(),"image_path":image_path.relative_to(ROOT).as_posix(),"labels":labels,"combo":combo,"split":split,"group_id":group,"duplicate_multiplicity":multiplicity})
        inventory.append({"combo":combo,"raw_after_code_filter":sum(x[2] for x in members),"unique_dihedral_groups":n,"splits":counts})
    rows.sort(key=lambda r:r["id"])
    write_jsonl(manifest,rows)
    pilot = []
    test_groups = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            test_groups[row["combo"]].append(row)
    for combo, members in sorted(test_groups.items()):
        pilot.extend(sorted(members,key=lambda r:r["group_id"])[:5])
    write_jsonl(out/"pilot.jsonl",pilot)
    validation_pilot=[]
    validation_groups=defaultdict(list)
    for row in rows:
        if row["split"]=="val":
            validation_groups[row["combo"]].append(row)
    for combo,members in sorted(validation_groups.items()):
        validation_pilot.extend(sorted(members,key=lambda r:r["group_id"])[:2])
    write_jsonl(out/"val_pilot.jsonl",validation_pilot)
    split_groups = {s:{r["group_id"] for r in rows if r["split"]==s} for s in ["train","val","test"]}
    assert not (split_groups["train"] & split_groups["test"] or split_groups["train"] & split_groups["val"] or split_groups["val"] & split_groups["test"])
    report = {"dataset":"MixedWM38","source_file":str(source),"source_sha256":sha256_file(source),"raw_shape":list(maps.shape),"label_order":LABELS,"raw_cardinality_counts":dict(sorted(Counter(targets.sum(axis=1).tolist()).items())),"unexpected_value_counts":{str(v):int((maps==v).sum()) for v in np.unique(maps) if v not in [0,1,2]},"quarantined_invalid_code_maps":int((~valid).sum()),"quarantined_label_conflict_groups":len(conflicts),"dihedral_groups_after_code_filter":len(groups),"selected_split_counts":dict(Counter(r["split"] for r in rows)),"pilot_samples":len(pilot),"seed":seed,"caps_per_combo":{"train":train_per_combo,"val":val_per_combo,"test":test_per_combo},"inventory":inventory,"leakage_checks":{"exact_rotated_mirrored_overlap":0,"policy":"one representative per D4 group; group split 70/15/15 by combination, then cap each split; no test-dependent tuning","limitation":"No lot or GAN parent IDs in this NPZ. D4 grouping cannot exclude shared synthetic parentage or pretraining contamination."},"provenance_caveat":"Author dataset includes generated/mixed maps; not an independent real-fab production benchmark."}
    write_json(out/"data_report.json",report)
    write_json(out/"quarantined_label_conflicts.json",conflicts)
    make_contact_sheet(pilot,out/"preview.png")
    return report


def make_contact_sheet(rows: list[dict], path: Path) -> None:
    by_combo = {}
    for row in rows:
        by_combo.setdefault(row["combo"],row)
    tiles = sorted(by_combo.items(),key=lambda pair:(len(pair[1]["labels"]),pair[0]))
    cols, width, height = 5, 215, 244
    canvas = Image.new("RGB",(cols*width,((len(tiles)+cols-1)//cols)*height),"white")
    draw = ImageDraw.Draw(canvas)
    for k,(combo,row) in enumerate(tiles):
        x,y=(k%cols)*width,(k//cols)*height
        with Image.open(resolve_path(row["image_path"])) as im:
            canvas.paste(im.resize((208,208),Image.Resampling.NEAREST),(x,y))
        # Captions are for this separate human-only preview, never API input.
        draw.text((x+2,y+210),combo[:30],fill="black")
        if len(combo)>30:
            draw.text((x+2,y+223),combo[30:60],fill="black")
    canvas.save(path)
