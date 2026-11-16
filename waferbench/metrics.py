"""Strict metrics for aligned, successful wafer-map predictions.

No failed request is converted into a prediction. The caller must report request
coverage separately and pass only successful, ground-truth-aligned records.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, hamming_loss,
    precision_recall_fscore_support,
)


def _label_set(value, allowed: set[str], context: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f"{context}: labels must be a list of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{context}: duplicate labels")
    if set(value) - allowed:
        raise ValueError(f"{context}: unknown labels {set(value) - allowed}")
    if "normal" in value and len(value) > 1:
        raise ValueError(f"{context}: normal cannot coexist with a defect label")
    return set(value) - {"normal"}


def _arrays(records: list[dict], labels: list[str], task: str):
    if task not in {"single", "multi"}:
        raise ValueError("task must be single or multi")
    if not records:
        raise ValueError("No successful predictions to evaluate")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be unique and nonempty")
    if task == "single" and "normal" not in labels:
        raise ValueError("single-label evaluation requires a normal class")
    if task == "multi" and "normal" in labels:
        raise ValueError("multi-label normal is an empty label set, not a label column")
    allowed = set(labels) | {"normal"}
    seen = set()
    truths, predictions, probabilities = [], [], []
    for i, row in enumerate(records):
        context = str(row.get("id", f"row {i}"))
        if not row.get("id") or row["id"] in seen:
            raise ValueError(f"{context}: missing or duplicate id")
        seen.add(row["id"])
        if row.get("status") not in {None, "ok", "success"} or row.get("error"):
            raise ValueError(f"{context}: failed requests cannot enter metric calculations")
        truth = _label_set(row["labels"], allowed, context)
        pred = _label_set(row["predicted_labels"], allowed, context)
        probs = row.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(labels):
            raise ValueError(f"{context}: probabilities must cover exactly {labels}")
        try:
            vector = np.asarray([probs[name] for name in labels], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid probability") from exc
        if not np.all(np.isfinite(vector)) or np.any((vector < 0) | (vector > 1)):
            raise ValueError(f"{context}: probabilities must be finite and in [0,1]")
        if task == "single":
            if len(truth) > 1 or len(pred) > 1:
                raise ValueError(f"{context}: multiple labels in single-label task")
            if not np.isclose(vector.sum(), 1, atol=0.005):
                raise ValueError(f"{context}: choice probabilities do not sum to one")
            truth_name = next(iter(truth), "normal")
            pred_name = next(iter(pred), "normal")
            if vector[labels.index(pred_name)] + 1e-9 < vector.max():
                raise ValueError(f"{context}: chosen label is not a maximum-probability label")
            truths.append(labels.index(truth_name))
            predictions.append(labels.index(pred_name))
        else:
            truths.append([int(name in truth) for name in labels])
            predictions.append([int(name in pred) for name in labels])
        probabilities.append(vector)
    return np.asarray(truths), np.asarray(predictions), np.asarray(probabilities)


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict:
    details, total = [], len(confidence)
    error = 0.0
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        selected = (confidence >= low) & ((confidence <= high) if i == bins - 1 else (confidence < high))
        count = int(selected.sum())
        if not count:
            continue
        acc, conf = float(correct[selected].mean()), float(confidence[selected].mean())
        error += count / total * abs(acc - conf)
        details.append({"lower": low, "upper": high, "n": count, "accuracy": acc, "confidence": conf})
    return {"ece_10_bins": float(error), "calibration_bins": details}


def _risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> list[dict]:
    # Include whole confidence-tie groups: arbitrary tie ordering must not change risk.
    order = np.argsort(-confidence, kind="stable")
    conf, error = confidence[order], 1 - correct[order].astype(float)
    cumulative = np.cumsum(error)
    endpoints = np.r_[np.flatnonzero(conf[:-1] != conf[1:]), len(conf) - 1]
    return [{"threshold": float(conf[i]), "coverage": float((i + 1) / len(conf)),
             "risk": float(cumulative[i] / (i + 1)), "accepted": int(i + 1)} for i in endpoints]


def evaluate_predictions(records: Iterable[dict], labels: list[str], task: str) -> dict:
    """Evaluate records. Empty defect-label sets denote normal wafers.

    Single-task Brier is the unscaled sum across classes (range 0..2).
    Multi-task Brier is the mean binary Brier across samples and labels.
    Multi-task sample F1 assigns 1 to correctly predicted empty normal sets.
    """
    records = list(records)
    y, pred, p = _arrays(records, labels, task)
    metrics = {"task": task, "n": len(records), "labels": labels,
               "label_cardinality_counts": dict(sorted(Counter(len(set(r["labels"]) - {"normal"}) for r in records).items()))}
    if task == "single":
        indices = np.arange(len(labels))
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=indices, zero_division=0)
        confidence = p.max(axis=1)
        correct = (y == pred)
        metrics.update({
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=indices, average="macro", zero_division=0)),
            "balanced_accuracy": float(np.mean(recall[np.asarray(support) > 0])),
            "confusion_matrix": confusion_matrix(y, pred, labels=indices).tolist(),
            "confusion_matrix_order": labels,
            "brier_score": float(np.mean(np.sum((p - np.eye(len(labels))[y]) ** 2, axis=1))),
            "brier_definition": "mean sum_k (p_k - one_hot_y_k)^2; range 0..2",
            "risk_coverage": _risk_coverage(confidence, correct),
            **_ece(confidence, correct),
        })
        if "normal" in labels:
            normal = labels.index("normal")
            defect_indices = [i for i in indices if i != normal]
            metrics["defect_only_macro_f1"] = float(f1_score(y, pred, labels=defect_indices, average="macro", zero_division=0))
    else:
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y, pred, average="micro", zero_division=0)
        correct = np.all(y == pred, axis=1)
        metrics.update({
            "exact_match_accuracy": float(correct.mean()),
            "subset_accuracy": float(correct.mean()),
            "micro_precision": float(precision_micro),
            "micro_recall": float(recall_micro),
            "micro_f1": float(f1_micro),
            "macro_f1": float(f1.mean()),
            "samples_f1": float(f1_score(y, pred, average="samples", zero_division=1)),
            "hamming_loss": float(hamming_loss(y, pred)),
            "brier_score": float(np.mean((p - y) ** 2)),
            "brier_definition": "mean_(sample,label) (P(label present) - y)^2; range 0..1",
        })
        for description, selected in (("single_or_normal", y.sum(axis=1) <= 1), ("mixed_only", y.sum(axis=1) >= 2)):
            if selected.any():
                metrics[description] = {
                    "n": int(selected.sum()),
                    "exact_match_accuracy": float(correct[selected].mean()),
                    "micro_f1": float(f1_score(y[selected], pred[selected], average="micro", zero_division=0)),
                    "macro_f1": float(f1_score(y[selected], pred[selected], average="macro", zero_division=0)),
                    "hamming_loss": float(hamming_loss(y[selected], pred[selected])),
                }
    metrics["per_class"] = {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                     "f1": float(f1[i]), "support": int(support[i])}
                            for i, label in enumerate(labels)}
    return metrics
