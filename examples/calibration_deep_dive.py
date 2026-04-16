"""
examples/calibration_deep_dive.py.
===================================
Deep-dive into calibration analysis with TrustLens.

Compares three models at different calibration levels:
* A well-calibrated model (Platt-scaled logistic regression)
* A slightly miscalibrated SVM
* A severely overconfident neural network (simulated)

Generates side-by-side reliability diagrams for visual comparison.

Run with:
  python examples/calibration_deep_dive.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from trustlens.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=2_000,
    n_features=15,
    n_informative=6,
    random_state=0,
    weights=[0.6, 0.4],
)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# 1. Well-calibrated logistic regression
lr = LogisticRegression(max_iter=500, random_state=0)
lr.fit(X_train, y_train)
prob_lr = lr.predict_proba(X_val)[:, 1]

# 2. Uncalibrated SVM (raw decision function passed through sigmoid)
svm_raw = SVC(kernel="rbf", probability=False, random_state=0)
svm_raw.fit(X_train, y_train)
# Hack: use calibration wrapper but get pre-calibration probs
svm_cal = CalibratedClassifierCV(
    SVC(kernel="rbf", probability=False, random_state=0), cv=5, method="sigmoid"
)
svm_cal.fit(X_train, y_train)
prob_svm = svm_cal.predict_proba(X_val)[:, 1]

# 3. Simulated overconfident model (push probs toward extremes)
prob_overconfident = np.clip(prob_lr**0.4, 0.01, 0.99)

models = [
    ("Logistic Regression (calibrated)", prob_lr),
    ("SVM + Platt Scaling", prob_svm),
    ("Simulated Overconfident Model", prob_overconfident),
]

# ---------------------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for ax, (name, probs) in zip(axes, models):
    frac_pos, mean_pred, _ = reliability_curve(y_val, probs)
    bs = brier_score(y_val, probs)
    ece = expected_calibration_error(y_val, probs)

    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1.5, label="Perfect")
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color="#F5784B")
    ax.plot(mean_pred, frac_pos, "o-", color="#4B8BF5", lw=2, markersize=7, label=name)

    ax.text(
        0.05,
        0.92,
        f"ECE = {ece:.3f}\nBS = {bs:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCC"),
        fontfamily="monospace",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)

plt.suptitle("Calibration Comparison — Three Models", fontsize=14, fontweight="bold")

import os

os.makedirs("examples/output", exist_ok=True)
fig.savefig("examples/output/calibration_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: examples/output/calibration_comparison.png")
plt.close(fig)
