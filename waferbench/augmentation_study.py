"""Train-only composition / feature ablation with validation-frozen fresh audit.

No external model or API is called. Existing test results are exploratory;
the previously unused mixed-only audit is evaluated after validation selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier

from .baseline import FEATURE_VERSION as BASE_VERSION
from .baseline import _predict_probabilities, spatial_features
from .common import LABELS, ROOT, read_jsonl, resolve_path, write_json, write_jsonl
from .data import dihedral_hash, render_map, sha256_file
from .metrics import evaluate_predictions

STUDY_VERSION="composition_features_v1"
PRIMARY_SEED=42
REPLICATION_SEEDS=[42,20261114,812]
PARAMETERS={"n_estimators":300,"min_samples_leaf":2,"max_features":"sqrt","class_weight":None,"n_jobs":-1}


def targets(rows):
    return np.asarray([[int(label in row["labels"]) for label in LABELS] for row in rows],dtype=np.uint8)


def records_from_probabilities(rows,probabilities,arm,seed):
    return [{"id":row["id"],"labels":row["labels"],"predicted_labels":[label for label,p in zip(LABELS,prob) if p>=.5],"probabilities":{label:float(p) for label,p in zip(LABELS,prob)},"split":row["split"],"task":"multi","backend":"spatial_random_forest_ablation","arm":arm,"seed":seed} for row,prob in zip(rows,probabilities)]


def metric_summary(records):
    result=evaluate_predictions(records,LABELS,"multi")
    active=[v["f1"] for v in result["per_class"].values() if v["support"]>0]
    result["macro_f1_over_labels_present"]=float(np.mean(active))
    mixed=[r for r in records if len(r["labels"])>=2]
    if mixed:
        mm=evaluate_predictions(mixed,LABELS,"multi")
        mm["macro_f1_over_labels_present"]=float(np.mean([v["f1"] for v in mm["per_class"].values() if v["support"]>0]))
        result["mixed_only"]=mm
    return result


def _selection_key(result):
    m=result["validation"]["mixed_only"]
    return (m["exact_match_accuracy"],m["micro_f1"],-result["n_features"],-result["training_samples"])


def _features(maps,version,path:Path,identity:dict):
    from .advanced_features import advanced_features, FEATURE_VERSION, feature_names
    extractor=spatial_features if version=="base" else advanced_features
    code=Path(__file__).with_name("baseline.py" if version=="base" else "advanced_features.py")
    metadata={"identity":identity,"version":BASE_VERSION if version=="base" else FEATURE_VERSION,"code_sha256":sha256_file(code),"n":len(maps)}
    if path.exists() and path.with_suffix(".json").exists():
        if json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))==metadata:
            with np.load(path,allow_pickle=False) as cache:
                matrix=cache["features"]
            expected=168 if version=="base" else len(feature_names())
            if matrix.shape==(len(maps),expected) and np.isfinite(matrix).all():
                return matrix
    start=time.monotonic()
    output=[]
    for i,wafer in enumerate(maps):
        output.append(extractor(wafer))
        if (i+1)%2000==0:
            print(f"Features {path.stem}: {i+1}/{len(maps)}",flush=True)
    matrix=np.stack(output)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite extracted features")
    np.savez_compressed(path,features=matrix)
    write_json(path.with_suffix(".json"),metadata)
    print(f"Features {path.stem}: {len(maps)} x {matrix.shape[1]}, {time.monotonic()-start:.1f}s",flush=True)
    return matrix


def _prepare_augmentations(train_rows,out:Path,n_per_combo:int):
    from .augmentation import generate_composites
    parent_ids={r["id"] for r in train_rows}
    generated={}
    audits={}
    for mode in ["naive_union","structured_mix"]:
        maps,y,provenance,audit=generate_composites(train_rows,n_per_combo=n_per_combo,seed=20261114,mode=mode)
        if len(maps)!=len(y) or len(y)!=len(provenance):
            raise ValueError("Augmentation output sizes do not match")
        if not len(maps):
            raise ValueError(f"No composites generated: {mode}")
        for p in provenance:
            # The generator also checks train-only parent membership internally.
            # Its complete schema is retained in the provenance file below.
            listed=p.get("parent_ids")
            if listed and not set(listed)<=parent_ids:
                raise ValueError("Augmentation parent outside the training set")
            for parent in p.get("parents",[]):
                if parent.get("source_id") not in parent_ids:
                    raise ValueError("Composite source parent outside the training set")
        generated[mode]=(maps,y,provenance)
        audits[mode]=audit
        write_json(out/f"{mode}_audit.json",audit)
        write_jsonl(out/f"{mode}_provenance.jsonl",provenance)
    def combo_counts(y):
        return Counter(tuple(int(b) for b in row) for row in y)
    a,b=[combo_counts(generated[mode][1]) for mode in ["naive_union","structured_mix"]]
    common={combo:min(count,b.get(combo,0)) for combo,count in a.items() if b.get(combo,0)>0}
    # Match added sample count and label-combination distribution for all arms.
    for mode,(maps,y,prov) in list(generated.items()):
        selected=[]
        for combo,n in sorted(common.items()):
            selected.extend([i for i,row in enumerate(y) if tuple(int(v) for v in row)==combo][:n])
        generated[mode]=(maps[selected],y[selected],[prov[i] for i in selected])
    rng=np.random.default_rng(20261114)
    lookup=defaultdict(list)
    for row in train_rows:
        if len(row["labels"])>=2:
            lookup[tuple(int(label in row["labels"]) for label in LABELS)].append(row)
    control_maps,control_y,control_prov=[],[],[]
    for combo,n in sorted(common.items()):
        pool=sorted(lookup[combo],key=lambda r:r["id"])
        for _ in range(n):
            row=pool[int(rng.integers(len(pool)))]
            wafer=np.load(resolve_path(row["map_path"]),allow_pickle=False)
            turns=int(rng.integers(4)); mirror=bool(rng.integers(2))
            transformed=np.rot90(wafer.T if mirror else wafer,turns).copy()
            control_maps.append(transformed); control_y.append(combo)
            control_prov.append({"parent_ids":[row["id"]],"parent_group_ids":[row["group_id"]],"parent_split":"train","turns":turns,"mirror":mirror})
    generated["d4_control"]=(np.stack(control_maps),np.asarray(control_y,dtype=np.uint8),control_prov)
    write_json(out/"matched_augmentation_counts.json",{"per_combo":[{"labels":[label for label,bit in zip(LABELS,combo) if bit],"n":n} for combo,n in sorted(common.items())],"per_arm_samples":sum(common.values()),"reason":"Equal added sample count and class-combination distribution across D4/union/structured arms; D4 is a transform/resampling control, not independent data."})
    for mode,(maps,y,provenance) in generated.items():
        np.savez_compressed(out/f"{mode}.npz",maps=maps,targets=y)
        write_jsonl(out/f"{mode}_matched_provenance.jsonl",provenance)
    return generated,audits


def _preview(generated,out:Path):
    # Human-only QA, never a model input.
    modes=["d4_control","naive_union","structured_mix"]
    h,w=244,220
    canvas=Image.new("RGB",(w*len(modes),h*4),"white")
    draw=ImageDraw.Draw(canvas)
    for col,mode in enumerate(modes):
        maps,y,_=generated[mode]
        for row,index in enumerate(np.linspace(0,len(maps)-1,4).astype(int)):
            path=out/"preview_tiles"/f"{mode}-{index}.png"
            render_map(maps[index],path,size=208)
            with Image.open(path) as im:
                canvas.paste(im,(col*w,row*h))
            draw.text((col*w+2,row*h+210),mode,fill="black")
            label="+".join(name for name,bit in zip(LABELS,y[index]) if bit)
            draw.text((col*w+2,row*h+224),label[:34],fill="black")
    canvas.save(out/"augmentation_preview.png")


def _fit(train_x,train_y,seed):
    model=RandomForestClassifier(**PARAMETERS,random_state=seed)
    model.fit(train_x,train_y)
    return model


def _block_importance(model,x,y,groups,out:Path):
    selected=y.sum(axis=1)>=2
    x,y=x[selected],y[selected]
    baseline=float(np.all((_predict_probabilities(model,x,"multi",LABELS)>=.5)==y,axis=1).mean())
    rng=np.random.default_rng(20261114)
    result={"split":"validation_mixed_only","n":len(x),"metric":"exact_match_accuracy","baseline":baseline,"interpretation":"Jointly permute a feature block; correlated groups can mask each other's contribution. Validation inspection only, no post-test feature selection.","groups":[]}
    for name,bounds in groups.items():
        start,stop=map(int,bounds)
        differences=[]
        for _ in range(3):
            changed=x.copy()
            changed[:,start:stop]=x[rng.permutation(len(x)),start:stop]
            probabilities=_predict_probabilities(model,changed,"multi",LABELS)
            score=float(np.all((probabilities>=.5)==y,axis=1).mean())
            differences.append(baseline-score)
        result["groups"].append({"name":name,"start":start,"stop":stop,"exact_match_drop_mean":float(np.mean(differences)),"drop_std":float(np.std(differences)),"repeats":differences})
    write_json(out,result)
    return result


def _paired_bootstrap(a_records,b_records):
    if [r["id"] for r in a_records]!=[r["id"] for r in b_records]:
        raise ValueError("Paired audit IDs differ")
    a=np.asarray([set(r["labels"])==set(r["predicted_labels"]) for r in a_records])
    b=np.asarray([set(r["labels"])==set(r["predicted_labels"]) for r in b_records])
    delta=b.astype(float)-a.astype(float)
    rng=np.random.default_rng(812)
    bootstrap=np.asarray([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)])
    return {"n":len(delta),"baseline_correct":int(a.sum()),"selected_correct":int(b.sum()),"delta_percentage_points":float(100*delta.mean()),"paired_bootstrap_95_percentage_points":(100*np.quantile(bootstrap,[.025,.975])).tolist(),"selected_only_correct":int((b&~a).sum()),"baseline_only_correct":int((a&~b).sum()),"both_correct":int((a&b).sum()),"caveat":"Map-level exploratory bootstrap; synthetic parent correlations are unknown and not represented."}


def run_study(out:Path=ROOT/"runs/augmentation_study",n_per_combo:int=100):
    from .advanced_features import FEATURE_GROUPS, FEATURE_VERSION, feature_names
    started=time.monotonic()
    out=resolve_path(out)
    if (out/"selection_frozen.json").exists():
        raise FileExistsError("A frozen experiment already exists here. Use another --out to avoid overwriting results.")
    out.mkdir(parents=True,exist_ok=True)
    cache=out/"feature_cache"; cache.mkdir(exist_ok=True)
    manifest=ROOT/"data/mixedwm38/manifest.jsonl"
    rows=read_jsonl(manifest)
    indices={s:np.asarray([i for i,r in enumerate(rows) if r["split"]==s]) for s in ["train","val","test"]}
    split_rows={s:[rows[i] for i in ix] for s,ix in indices.items()}
    group_sets={s:{r["group_id"] for r in split_rows[s]} for s in split_rows}
    if any(group_sets[a]&group_sets[b] for a,b in [("train","val"),("train","test"),("val","test")]):
        raise ValueError("Group leakage in reference manifest")
    config={"study_version":STUDY_VERSION,"primary_seed":PRIMARY_SEED,"replication_seeds":REPLICATION_SEEDS,"parameters":PARAMETERS,"decoder":"fixed 0.5 for every label and arm; no threshold tuning","selection":"validation mixed-only exact match, then micro F1; ties fewer features/samples","reference_manifest_sha256":sha256_file(manifest),"advanced_feature_version":FEATURE_VERSION,"advanced_feature_groups":FEATURE_GROUPS,"n_per_combo_requested":n_per_combo,"prior_test_visibility":"Original capped test has been inspected in earlier work. All results on that test are exploratory. A previously unused original-test-pool mixed-only audit is opened only after validation selection is frozen.","external_api_calls":0,"date_local":datetime.now(timezone(timedelta(hours=8))).isoformat()}
    write_json(out/"config.json",config)
    print("Loading original maps / constructing train-only augmentations",flush=True)
    maps=[np.load(resolve_path(r["map_path"]),allow_pickle=False) for r in rows]
    y=targets(rows)
    augmentation,audits=_prepare_augmentations(split_rows["train"],out,n_per_combo)
    generated_groups={mode:{dihedral_hash(wafer) for wafer in generated[0]} for mode,generated in augmentation.items()}
    prior_evaluation_groups=group_sets["val"]|group_sets["test"]
    if any(groups&prior_evaluation_groups for groups in generated_groups.values()):
        raise ValueError("Generated training image exactly matches a reference evaluation image")
    write_json(out/"augmentation_group_audit.json",{mode:{"n":len(augmentation[mode][0]),"unique_d4_groups":len(groups),"reference_validation_or_test_overlap":len(groups&prior_evaluation_groups)} for mode,groups in generated_groups.items()})
    _preview(augmentation,out)
    original_x={}
    augmented_x={}
    for version in ["base","advanced"]:
        original_x[version]=_features(maps,version,cache/f"original_{version}.npz",{"manifest_sha256":sha256_file(manifest)})
        for mode,(amaps,ay,prov) in augmentation.items():
            augmented_x[version,mode]=_features(amaps,version,cache/f"{mode}_{version}.npz",{"augmentation_sha256":sha256_file(out/f"{mode}.npz")})
    results=[]
    trained={}
    training_data={}
    def arm_run(name,version,mode,singles_only=False):
        base_indices=indices["train"]
        if singles_only:
            base_indices=np.asarray([i for i in base_indices if len(rows[i]["labels"])<=1])
        train_x,train_y=original_x[version][base_indices],y[base_indices]
        if mode!="none":
            train_x=np.concatenate([train_x,augmented_x[version,mode]])
            train_y=np.concatenate([train_y,augmentation[mode][1]])
        model=_fit(train_x,train_y,PRIMARY_SEED)
        validation=records_from_probabilities(split_rows["val"],_predict_probabilities(model,original_x[version][indices["val"]],"multi",LABELS),name,PRIMARY_SEED)
        result={"arm":name,"features":version,"augmentation":mode,"singles_only_original_training":singles_only,"n_features":train_x.shape[1],"training_samples":len(train_x),"validation":metric_summary(validation),"candidate_for_primary_selection":not singles_only}
        if name=="base_none":
            historical=ROOT/"runs/mixedwm38/rf_multi/val_predictions.jsonl"
            if historical.exists():
                old_val=read_jsonl(historical)
                current={r["id"]:r for r in validation}
                if {r["id"] for r in old_val}!=set(current):
                    raise ValueError("Reference baseline validation sample IDs changed")
                delta=max(abs(row["probabilities"][label]-current[row["id"]]["probabilities"][label]) for row in old_val for label in LABELS)
                write_json(out/"baseline_reproduction.json",{"status":"compared_optional_historical_run","same_validation_ids":True,"max_absolute_probability_difference":delta,"fixed_decoder":0.5})
                if delta>1e-9:
                    raise ValueError("The reference baseline probabilities did not reproduce")
            else:
                write_json(out/"baseline_reproduction.json",{"status":"fresh_clone_no_historical_run","policy":"The base_none arm is fitted directly from this repository's train split; no external run is required.","fixed_decoder":0.5})
        directory=out/name; directory.mkdir(exist_ok=True)
        write_jsonl(directory/"validation_predictions.jsonl",validation)
        write_json(directory/"validation_metrics.json",result)
        results.append(result); trained[name]=model; training_data[name]=(train_x,train_y)
        print(f"Arm {name}: validation mixed exact={result['validation']['mixed_only']['exact_match_accuracy']:.4f}, microF1={result['validation']['mixed_only']['micro_f1']:.4f}",flush=True)
    for version in ["base","advanced"]:
        for mode in ["none","d4_control","naive_union","structured_mix"]:
            arm_run(f"{version}_{mode}",version,mode)
    # Separate question: can compositions help when no supplied mixed training
    # maps are used? These two arms do not enter the main model selection.
    arm_run("advanced_single_only","advanced","none",True)
    arm_run("advanced_single_plus_structured","advanced","structured_mix",True)
    candidates=[r for r in results if r["candidate_for_primary_selection"]]
    selected=max(candidates,key=_selection_key)
    frozen={"selected_arm":selected["arm"],"selection_uses":"validation predictions only; original/fresh test predictions not computed yet","frozen_at_local":datetime.now(timezone(timedelta(hours=8))).isoformat(),"selection_rule":config["selection"],"validation_ranking":[{"arm":r["arm"],"mixed_exact":r["validation"]["mixed_only"]["exact_match_accuracy"],"mixed_micro_f1":r["validation"]["mixed_only"]["micro_f1"]} for r in sorted(candidates,key=_selection_key,reverse=True)],"fresh_audit_evaluation":"Baseline and validation-selected arm only; seeds fixed before audit. No adaptation after audit."}
    write_json(out/"selection_frozen.json",frozen)
    # Only now compute old-test scores and read unused fresh maps.
    for result in results:
        name=result["arm"]; version=result["features"]
        records=records_from_probabilities(split_rows["test"],_predict_probabilities(trained[name],original_x[version][indices["test"]],"multi",LABELS),name,PRIMARY_SEED)
        result["old_test_exploratory"]=metric_summary(records)
        write_jsonl(out/name/"test_predictions.jsonl",records)
        write_json(out/name/"metrics.json",result)
    fresh_manifest=ROOT/"data/mixedwm38_fresh_mixed/manifest.jsonl"
    if not fresh_manifest.exists():
        from .fresh_holdout import prepare_fresh_mixed_holdout
        prepare_fresh_mixed_holdout()
    fresh_rows=read_jsonl(fresh_manifest)
    fresh_ids={r["id"] for r in fresh_rows}
    if fresh_ids&{r["id"] for r in rows}:
        raise ValueError("Fresh audit overlaps previously exported reference samples")
    fresh_groups={r["group_id"] for r in fresh_rows}
    if any(groups&fresh_groups for groups in generated_groups.values()):
        raise ValueError("Generated training image exactly matches a fresh audit image; refusing evaluation")
    write_json(out/"fresh_augmentation_overlap_audit.json",{"id_overlap_with_reference":0,"generated_d4_overlap":{mode:len(groups&fresh_groups) for mode,groups in generated_groups.items()}})
    fresh_maps=[np.load(resolve_path(r["map_path"]),allow_pickle=False) for r in fresh_rows]
    fresh_x={version:_features(fresh_maps,version,cache/f"fresh_{version}.npz",{"manifest_sha256":sha256_file(fresh_manifest)}) for version in {"base",selected["features"]}}
    fresh_results=[]
    fresh_primary={}
    for name in dict.fromkeys(["base_none",selected["arm"]]):
        arm=next(r for r in results if r["arm"]==name)
        version=arm["features"]
        seed_results=[]
        for seed in REPLICATION_SEEDS:
            model=trained[name] if seed==PRIMARY_SEED else _fit(*training_data[name],seed)
            records=records_from_probabilities(fresh_rows,_predict_probabilities(model,fresh_x[version],"multi",LABELS),name,seed)
            metrics=metric_summary(records)
            write_jsonl(out/name/f"fresh_predictions_seed_{seed}.jsonl",records)
            write_json(out/name/f"fresh_metrics_seed_{seed}.json",metrics)
            seed_results.append({"seed":seed,"metrics":metrics})
            if seed==PRIMARY_SEED:
                fresh_primary[name]=records
        fresh_results.append({"arm":name,"seeds":seed_results,"exact_match_mean":float(np.mean([r["metrics"]["exact_match_accuracy"] for r in seed_results])),"exact_match_std":float(np.std([r["metrics"]["exact_match_accuracy"] for r in seed_results])),"micro_f1_mean":float(np.mean([r["metrics"]["micro_f1"] for r in seed_results]))})
    paired=_paired_bootstrap(fresh_primary["base_none"],fresh_primary[selected["arm"]])
    # Inspect the selected model on validation, never on the fresh audit.
    selected_groups=FEATURE_GROUPS if selected["features"]=="advanced" else {"base168":(0,168)}
    importance=_block_importance(trained[selected["arm"]],original_x[selected["features"]][indices["val"]],y[indices["val"]],selected_groups,out/"validation_group_importance.json")
    selected_metadata={"task":"multi","classes":LABELS,"feature_set":selected["features"],
                       "feature_version":FEATURE_VERSION if selected["features"]=="advanced" else BASE_VERSION,
                       "n_features":selected["n_features"],"thresholds":[.5]*len(LABELS),
                       "seed":PRIMARY_SEED,"arm":selected["arm"],"parameters":PARAMETERS}
    joblib.dump({"model":trained[selected["arm"]],**selected_metadata},out/"selected_model.joblib",compress=3)
    write_json(out/"selected_model_metadata.json",selected_metadata)
    write_json(out/"feature_names.json",feature_names() if selected["features"]=="advanced" else feature_names()[:168])
    write_json(out/"fresh_evaluation_receipt.json",{"evaluated_at_local":datetime.now(timezone(timedelta(hours=8))).isoformat(),"prepared_manifest_sha256":sha256_file(fresh_manifest),"selection_frozen_sha256":sha256_file(out/"selection_frozen.json"),"evaluated_arms":list(fresh_primary),"training_seeds":REPLICATION_SEEDS,"n":len(fresh_rows),"status":"completed_once_after_validation_selection"})
    summary={"config":config,"selection":frozen,"arms":results,"fresh_audit":fresh_results,
             "fresh_audit_protocol":{"n":len(fresh_rows),"combinations":len({tuple(r['labels']) for r in fresh_rows}),"sources":sorted({r.get('source','unspecified') for r in fresh_rows}),"manifest_sha256":sha256_file(fresh_manifest)},
             "paired_fresh_audit_seed42":paired,"validation_feature_group_importance":importance,"elapsed_seconds":time.monotonic()-started,"caveats":["MixedWM38 contains generated maps; unknown source parentage prevents independent real-fab claims.","A same-source audit is not a new industrial domain; consult the preparation audit for exact provenance.","The OR/structure heuristics produce assumed labels; physical validity and pixel-level signal masks are not established.","Primary decoder fixed0.5; prior 70.9% used validation-F1 thresholds, so matched fixed0.5 reference is about71.9% mixed exact.","Only baseline and validation-selected arm evaluated on fresh audit; all old-test ablations explicitly exploratory."]}
    write_json(out/"summary.json",summary)
    write_report(summary,out)
    print(json.dumps({"selected":selected["arm"],"fresh_audit":paired,"elapsed_seconds":summary["elapsed_seconds"],"report":str(out/"report.md")},ensure_ascii=False,indent=2),flush=True)
    return summary


def write_report(summary,out):
    def pct(v):return f"{100*v:.1f}%"
    names={"none":"原训练集","d4_control":"旋转/镜像增量对照","naive_union":"单模式直接并集","structured_mix":"单模式结构组合+一次背景"}
    ntrees=summary['config']['parameters']['n_estimators']
    lines=["# RF：单模式组合增强与新增特征消融","","本报告只评估图像级分类，没有评估CAM、逐pattern mask或图案还原。","",f"所有增强父样本来自原训练集。训练/验证分区不变；统一{ntrees}棵RF、随机种子42和固定0.5阈值。原测试集的全部消融成绩均标为探索性。","","## 验证选择与旧测试消融","","| 特征 | 增强方式 | 训练数量 | 验证混合整组正确率 | 旧测试混合整组正确率 | 旧测试混合Micro F1 |","|---|---|---:|---:|---:|---:|"]
    for r in summary["arms"]:
        if not r["candidate_for_primary_selection"]:continue
        v=r["validation"]["mixed_only"]; t=r["old_test_exploratory"]["mixed_only"]
        lines.append(f"| {r['n_features']}维 | {names[r['augmentation']]} | {r['training_samples']} | {pct(v['exact_match_accuracy'])} | {pct(t['exact_match_accuracy'])} | {pct(t['micro_f1'])} |")
    chosen=summary["selection"]["selected_arm"]
    audit=summary['fresh_audit_protocol']
    lines += ["",f"按验证集预先冻结的规则选择 **{chosen}**，选择记录写入后才评估审计集。没有根据审计集重新挑模型。","","## 混合审计集","",f"本次审计集包含{audit['combinations']}种组合、{audit['n']}张图；来源字段为{', '.join(audit['sources'])}。ID/D4交叉检查已通过，完整来源与原分区重放情况以准备阶段的report.json为准。程序化smoke夹具的分数不作为研究结果。","","| 模型 | seed42整组正确率 | 3个训练种子平均整组正确率 | 种子间标准差 | 平均Micro F1 |","|---|---:|---:|---:|---:|"]
    for r in summary["fresh_audit"]:
        m=r["seeds"][0]["metrics"]
        lines.append(f"| {r['arm']} | {pct(m['exact_match_accuracy'])} | {pct(r['exact_match_mean'])} | {100*r['exact_match_std']:.2f}个百分点 | {pct(r['micro_f1_mean'])} |")
    paired=summary["paired_fresh_audit_seed42"]
    lo,hi=paired["paired_bootstrap_95_percentage_points"]
    lines += ["",f"seed42配对提升 **{paired['delta_percentage_points']:+.2f}个百分点**，按图bootstrap的95%区间 [{lo:+.2f}, {hi:+.2f}] 个百分点。只改进{paired['selected_only_correct']}张、只退步{paired['baseline_only_correct']}张。此区间不能排除未知合成父样本相关性。","","## 只用单模式原图训练的补充对照","","这两组不使用原数据集中给出的混合训练图像像素，但合法组合词表仍来自混合训练标签，因此不能称为完全没有混合监督或未知组合零样本泛化。它们不参与主方案选择，检验合成组合对单模式向混合识别迁移的帮助。","","| 方案 | 训练数量 | 验证混合整组正确率 | 旧测试混合整组正确率 | 旧测试混合Micro F1 |","|---|---:|---:|---:|---:|"]
    for r in summary["arms"]:
        if r["candidate_for_primary_selection"]:continue
        v,t=r["validation"]["mixed_only"],r["old_test_exploratory"]["mixed_only"]
        lines.append(f"| {r['arm']} | {r['training_samples']} | {pct(v['exact_match_accuracy'])} | {pct(t['exact_match_accuracy'])} | {pct(t['micro_f1'])} |")
    lines += ["","## 新增特征与解释边界","","- 原168维保留。新增径向×角向联合分布、背景标准化直线投影、多尺度局部密度及连通域形状；具体分组和维度见feature_names.json/config.json。","- 直接并集会叠加背景噪声；结构组合使用启发式信号软筛选后只补一次背景。保留完整父样本、变换和拒绝记录，无法保证每张合成图的物理正确性或所有组成模式仍可见。","- 三种增加数据的主实验臂匹配增加的样本数量与标签组合分布；旋转/镜像是重采样控制，不是新增独立晶圆。","- 没有对新审计结果继续调参，也没有将测试标签用于增强或特征计算。","- Random和Near-full未被强行混入训练集原本不存在的标签组合，因此没有验证这些类别的新组合泛化。","- 全部实验在本地运行，没有调用大模型API。","","## 验证集特征分组置乱","","分组置乱反映所选RF对特征的依赖，不代表独立因果贡献；相关组可以相互替代。","","| 特征组 | 置乱后整组正确率平均下降 |","|---|---:|"]
    for g in sorted(summary["validation_feature_group_importance"]["groups"],key=lambda g:g["exact_match_drop_mean"],reverse=True):
        lines.append(f"| {g['name']} | {100*g['exact_match_drop_mean']:+.2f}个百分点 |")
    lines += ["","![消融比较](ablation.png)","","![组合增强预览，仅人工检查](augmentation_preview.png)","","来源：[MixedWM38原作者](https://github.com/Junliangwangdhu/WaferMap)、[RF官方实现](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)、[置乱重要性说明](https://scikit-learn.org/stable/modules/permutation_importance.html)。",""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
    main=[r for r in summary["arms"] if r["candidate_for_primary_selection"]]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    modes=["none","d4_control","naive_union","structured_mix"]
    display=["Original","D4 control","Naive union","Structure mix"]
    for ax,key,title in [(axes[0],"validation","Validation: mixed-only exact match"),(axes[1],"old_test_exploratory","Previously seen test: exploratory")]:
        x=np.arange(4)
        for offset,version,color in [(-.18,"base","#355d99"),(.18,"advanced","#dc8c37")]:
            rr=[next(r for r in main if r["features"]==version and r["augmentation"]==mode) for mode in modes]
            values=[r[key]["mixed_only"]["exact_match_accuracy"] for r in rr]
            bars=ax.bar(x+offset,values,.36,label=f"{rr[0]['n_features']} features",color=color)
            ax.bar_label(bars,labels=[f"{100*v:.1f}" for v in values],fontsize=8,padding=3)
        ax.set_xticks(x,display,rotation=12); ax.set_ylim(0,1); ax.set_title(title); ax.set_ylabel("Exact match"); ax.grid(axis="y",alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(out/"ablation.png",dpi=170); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"runs/augmentation_study")
    parser.add_argument("--n-per-combo",type=int,default=100)
    args=parser.parse_args()
    run_study(args.out,args.n_per_combo)
