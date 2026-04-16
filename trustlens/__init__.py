"""
TrustLens — Debug your ML models beyond accuracy.

A modular, research-backed Python library for deep model analysis:
 - Calibration (Brier Score, ECE, reliability diagrams)
 - Failure Analysis (misclassifications, confidence gaps)
 - Explainability (Grad-CAM, Eigen-CAM)
 - Faithfulness Testing (pixel deletion/insertion)
 - Bias Detection (subgroup performance, class imbalance)
 - Representation Analysis (embedding geometry, CKA)
 - Trust Score (0–100 composite trustworthiness metric)
"""

__version__ = "0.1.0"
__author__ = "Shahid Ul Islam"
__license__ = "MIT"

from trustlens.api import analyze, quick_analyze
from trustlens.report import TrustReport
from trustlens.trust_score import TrustScoreResult, compute_trust_score

__all__ = [
    "analyze",
    "quick_analyze",
    "TrustReport",
    "compute_trust_score",
    "TrustScoreResult",
    "__version__",
]
