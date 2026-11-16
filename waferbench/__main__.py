"""RF-only command line: no network credentials or cloud models required."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from .common import LABELS, SINGLE_LABELS, ROOT, read_jsonl, resolve_path, write_json


def main():
    parser=argparse.ArgumentParser(description="Wafer-map random-forest classification")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="Check local Python dependencies")
    prepare=sub.add_parser("prepare",help="Prepare MixedWM38 with D4 grouping")
    prepare.add_argument("--source",type=Path,default=Path("data_sources/Wafer_Map_Datasets.npz"))
    prepare.add_argument("--out",type=Path,default=Path("data/mixedwm38"))
    prepare.add_argument("--seed",type=int,default=42)
    for name,default in [("train",200),("val",20),("test",20)]:
        prepare.add_argument(f"--{name}-per-combo",type=int,default=default)
    wm=sub.add_parser("prepare-wm811k",help="Prepare WM-811K using lot-disjoint splits")
    wm.add_argument("--source",type=Path,default=Path("data_sources/LSWMD.pkl"))
    wm.add_argument("--out",type=Path,default=Path("data/wm811k"))
    wm.add_argument("--seed",type=int,default=42)
    for name,default in [("train",200),("val",20),("test",20)]:
        wm.add_argument(f"--{name}-per-class",type=int,default=default)
    fresh=sub.add_parser("fresh-holdout",help="Replay the default MixedWM38 split and reserve an unused mixed audit")
    fresh.add_argument("--source",type=Path,default=Path("data_sources/Wafer_Map_Datasets.npz"))
    fresh.add_argument("--reference-manifest",type=Path,default=Path("data/mixedwm38/manifest.jsonl"))
    fresh.add_argument("--out",type=Path,default=Path("data/mixedwm38_fresh_mixed"))
    fresh.add_argument("--seed",type=int,default=42)
    fresh.add_argument("--per-combo",type=int,default=20)
    baseline=sub.add_parser("baseline",help="Train RF, select thresholds on validation and evaluate test")
    baseline.add_argument("--manifest",type=Path,default=Path("data/mixedwm38/manifest.jsonl"))
    baseline.add_argument("--task",choices=["single","multi"],required=True)
    baseline.add_argument("--features",choices=["base","advanced"],default="base")
    baseline.add_argument("--seed",type=int,default=42)
    baseline.add_argument("--out",type=Path,required=True)
    study=sub.add_parser("augmentation-study",help="Run the fixed composition/feature ablation")
    study.add_argument("--out",type=Path,default=Path("runs/augmentation_study"))
    study.add_argument("--n-per-combo",type=int,default=100)
    evaluate=sub.add_parser("evaluate",help="Evaluate existing prediction JSONL")
    evaluate.add_argument("--predictions",type=Path,required=True)
    evaluate.add_argument("--task",choices=["single","multi"],required=True)
    evaluate.add_argument("--out",type=Path)
    feature=sub.add_parser("features",help="Extract 168 or 502 numerical features from a 0/1/2 map")
    feature.add_argument("--map",type=Path,required=True)
    feature.add_argument("--features",choices=["base","advanced"],default="base")
    feature.add_argument("--out",type=Path)
    predict=sub.add_parser("predict",help="Predict using a model bundle trained by this repository")
    predict.add_argument("--model",type=Path,required=True)
    predict.add_argument("--map",type=Path,required=True)
    predict.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.command=="doctor":
        versions={}
        for name in ["numpy","scipy","scikit-learn","Pillow","matplotlib","joblib","pandas"]:
            try: versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name]="MISSING"
        result={"python":platform.python_version(),"project":str(ROOT),"packages":versions}
    elif args.command=="prepare":
        from .data import prepare_mixed
        result=prepare_mixed(args.source,args.out,args.seed,args.train_per_combo,args.val_per_combo,args.test_per_combo)
    elif args.command=="prepare-wm811k":
        from .wm811k import prepare_wm811k
        result=prepare_wm811k(args.source,args.out,args.seed,args.train_per_class,args.val_per_class,args.test_per_class)
    elif args.command=="fresh-holdout":
        from .fresh_holdout import prepare_fresh_mixed_holdout
        result=prepare_fresh_mixed_holdout(args.source,args.reference_manifest,args.out,args.seed,args.per_combo)
    elif args.command=="baseline":
        from .baseline import run_baseline
        result=run_baseline(resolve_path(args.manifest),resolve_path(args.out),args.task,args.seed,args.features)
    elif args.command=="augmentation-study":
        from .augmentation_study import run_study
        result=run_study(resolve_path(args.out),args.n_per_combo)
    elif args.command=="evaluate":
        from .metrics import evaluate_predictions
        result=evaluate_predictions(read_jsonl(resolve_path(args.predictions)),SINGLE_LABELS if args.task=="single" else LABELS,args.task)
        if args.out: write_json(resolve_path(args.out),result)
    elif args.command in {"features","predict"}:
        import numpy as np
        from .baseline import spatial_features, _predict_probabilities
        from .advanced_features import advanced_features, feature_names
        wafer=np.load(resolve_path(args.map),allow_pickle=False)
        if args.command=="features":
            vector=spatial_features(wafer) if args.features=="base" else advanced_features(wafer)
            names=[f"spatial_v1_{i:03d}" for i in range(168)] if args.features=="base" else feature_names()
            result={"feature_set":args.features,"n_features":len(vector),"names":names,"values":vector.tolist()}
        else:
            import joblib
            bundle=joblib.load(resolve_path(args.model))
            vector=spatial_features(wafer) if bundle["feature_set"]=="base" else advanced_features(wafer)
            probabilities=_predict_probabilities(bundle["model"],vector[None,:],bundle["task"],bundle["classes"])[0]
            if bundle["task"]=="single":
                winner=bundle["classes"][int(probabilities.argmax())]
                predicted=[] if winner=="normal" else [winner]
            else:
                predicted=[name for name,p,threshold in zip(bundle["classes"],probabilities,bundle["thresholds"]) if p>=threshold]
            result={"task":bundle["task"],"feature_set":bundle["feature_set"],"predicted_labels":predicted,"probabilities":{k:float(p) for k,p in zip(bundle["classes"],probabilities)}}
        if args.out: write_json(resolve_path(args.out),result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
