"""
trustlens.metrics.bias.
=======================
Bias and fairness detection.

Bias in ML manifests as systematically worse performance for certain
subgroups (demographic, geographic, temporal, etc.). TrustLens surfaces
these disparities without making causal claims — the responsibility to
act lies with the practitioner.

Metrics implemented
-------------------
* ``class_imbalance_report`` — distribution statistics for label classes.
* ``subgroup_performance``  — per-subgroup accuracy/F1 breakdown.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def class_imbalance_report(y_true: np.ndarray) -> dict:
    """
    Summarize the class distribution in ``y_true``.

    Reports absolute counts, relative frequencies, and an imbalance
    ratio (majority class count / minority class count).

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth labels.

    Returns
    -------
    dict with keys:
      * ``class_counts``   — dict mapping class → sample count
      * ``class_frequencies`` — dict mapping class → relative frequency
      * ``imbalance_ratio`` — max_count / min_count (1.0 = perfectly balanced)
      * ``minority_class``  — class with fewest samples
      * ``majority_class``  — class with most samples

    Examples
    --------
    >>> report = class_imbalance_report(y_true)
    >>> print(f"Imbalance ratio: {report['imbalance_ratio']:.2f}x")
    """
    y_true = np.asarray(y_true)
    classes, counts = np.unique(y_true, return_counts=True)
    n = len(y_true)

    class_counts = {int(cls): int(cnt) for cls, cnt in zip(classes, counts)}
    class_frequencies = {int(cls): round(float(cnt / n), 4) for cls, cnt in zip(classes, counts)}

    min_count = int(counts.min())
    max_count = int(counts.max())
    minority_class = int(classes[counts.argmin()])
    majority_class = int(classes[counts.argmax()])
    imbalance_ratio = round(max_count / min_count, 4) if min_count > 0 else float("inf")

    return {
        "class_counts": class_counts,
        "class_frequencies": class_frequencies,
        "imbalance_ratio": imbalance_ratio,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "n_classes": int(len(classes)),
    }


def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: dict[str, np.ndarray],
    metrics: Optional[list[str]] = None,
) -> dict:
    """
    Compute model performance broken down by sensitive subgroups.

    For each feature in ``sensitive_features``, TrustLens computes
    per-group accuracy and macro-F1 scores, then derives the
    *performance gap* between best and worst performing groups.

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth labels.
    y_pred : np.ndarray
      Model predictions.
    sensitive_features : dict
      Mapping of feature name → 1-D array of group labels.
      Example: ``{"gender": gender_array}``.
    metrics : list[str], optional
      Which metrics to compute. Supports ``"accuracy"`` and ``"f1"``.
      Default: ``["accuracy", "f1"]``.

    Returns
    -------
    dict
      Nested dict: feature → group → metric values + summary.

    Examples
    --------
    >>> results = subgroup_performance(
    ...   y_true, y_pred,
    ...   sensitive_features={"gender": gender_array},
    ... )
    >>> print(results["gender"]["performance_gap"])
    """
    if metrics is None:
        metrics = ["accuracy", "f1"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report: dict = {}

    for feature_name, group_array in sensitive_features.items():
        group_array = np.asarray(group_array)
        groups = np.unique(group_array)
        group_results: dict = {}

        for g in groups:
            mask = group_array == g
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]

            group_metrics: dict = {"n_samples": int(mask.sum())}

            if "accuracy" in metrics:
                group_metrics["accuracy"] = round(float(accuracy_score(y_true_g, y_pred_g)), 4)
            if "f1" in metrics:
                group_metrics["f1"] = round(
                    float(
                        f1_score(
                            y_true_g,
                            y_pred_g,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                )

            group_results[str(g)] = group_metrics

        # Compute performance gap (accuracy-based)
        if "accuracy" in metrics and len(group_results) >= 2:
            accuracies = [v["accuracy"] for v in group_results.values()]
            gap = round(max(accuracies) - min(accuracies), 4)
            best_group = max(group_results, key=lambda g: group_results[g].get("accuracy", 0))
            worst_group = min(group_results, key=lambda g: group_results[g].get("accuracy", 0))
            group_results["__summary__"] = {
                "performance_gap": gap,
                "best_group": best_group,
                "worst_group": worst_group,
            }

        report[feature_name] = group_results

    return report
