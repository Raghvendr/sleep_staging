from __future__ import annotations

import numpy as np


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def cohen_kappa_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix_numpy(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    total = cm.sum()
    if total == 0:
        return 0.0
    observed = np.trace(cm) / total
    row_marginals = cm.sum(axis=1) / total
    col_marginals = cm.sum(axis=0) / total
    expected = float((row_marginals * col_marginals).sum())
    if expected == 1.0:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: list[str],
) -> dict[str, dict[str, float]]:
    num_classes = len(stage_names)
    cm = confusion_matrix_numpy(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    report: dict[str, dict[str, float]] = {}
    for idx, stage_name in enumerate(stage_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[idx, :].sum())
        report[stage_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(support),
        }
    return report


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: list[str],
) -> dict[str, object]:
    accuracy = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    num_classes = len(stage_names)
    confusion = confusion_matrix_numpy(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    kappa = cohen_kappa_numpy(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    report = classification_report_dict(y_true=y_true, y_pred=y_pred, stage_names=stage_names)
    macro_f1 = float(np.mean([values["f1"] for values in report.values()])) if report else 0.0

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "per_class": report,
    }
