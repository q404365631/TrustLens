"""
trustlens.trust_score.
======================
The TrustLens Trust Score — a single 0–100 composite measure of model
trustworthiness.

Why a single score?
-------------------
Practitioners face "metric overload": ECE, Brier Score, silhouette scores,
confidence gaps — great individually but hard to act on as a whole.

The Trust Score distils all TrustLens analysis into one instantly readable
number:

 * **< 40** — Serious issues. Do not deploy.
 * **40–60** — Moderate trust. Investigate flagged dimensions.
 * **60–80** — Good. Minor improvements recommended.
 * **80–100** — High trust. Model is production-ready.

Formula
-------
The Trust Score is a weighted sum of four normalized sub-scores (0–100 each):

 TrustScore = w_cal * CalibrationScore
       + w_fail * FailureScore
       + w_bias * BiasScore
       + w_rep * RepresentationScore

Default weights (tuned to reflect deployment risk):
 w_cal = 0.35  (calibration matters most — drives overconfidence risk)
 w_fail = 0.30  (failure patterns drive safety risk)
 w_bias = 0.25  (bias drives fairness/regulatory risk)
 w_rep = 0.10  (representation is a bonus signal; not always available)

If a dimension is unavailable (e.g., no embeddings → no representation score),
its weight is redistributed proportionally to the other available dimensions.

Sub-score Normalization
-----------------------
All sub-scores are normalized to [0, 100]:

 * CalibrationScore = 100 × (1 - clip(0.5×BS + 0.5×ECE, 0, 1))
   - Brier Score and ECE are both in [0, 1]; lower is better.
   - Perfect calibration → 100. Worst case (BS=1, ECE=1) → 0.

 * FailureScore = 100 × clip(confidence_gap, 0, 1)
   - Confidence gap in [0, 1] (clipped); higher is better.
   - A model that is highly confident *only* when correct → 100.

 * BiasScore = 100 × (1 - clip(bias_penalty, 0, 1))
   - bias_penalty = 0.5 × clip(imbalance_ratio / 20, 0, 1)
           + 0.5 × clip(subgroup_gap, 0, 1)
   - Perfectly balanced dataset, zero subgroup gap → 100.

 * RepresentationScore = 100 × clip(0.5 + 0.5 × silhouette, 0, 1)
   - Silhouette ∈ [-1, 1]; mapped to [0, 100].
   - Perfect separation → 100. Total overlap → 0.

References
----------
* Brier (1950), Guo et al. (2017) — calibration
* Hardt et al. (2016) — fairness
* Rousseeuw (1987) — silhouette
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "calibration": 0.35,
    "failure": 0.30,
    "bias": 0.25,
    "representation": 0.10,
}

_GRADE_THRESHOLDS = [
    (80, "A", "High Trust - production-ready"),
    (60, "B", "Good Trust - minor issues to address"),
    (40, "C", "Moderate Trust - investigate flagged dimensions"),
    (0, "D", "Low Trust - serious issues, do not deploy"),
]


# ---------------------------------------------------------------------------
# Sub-score computers
# ---------------------------------------------------------------------------


def _calibration_score(cal_data: dict) -> float:
    """
    Compute calibration sub-score (0–100).

    CalibScore = 100 × (1 − clip(0.5×BS + 0.5×ECE, 0, 1))
    """
    bs = float(cal_data.get("brier_score", 0.5))
    ece = float(cal_data.get("ece", 0.5))
    composite = 0.5 * bs + 0.5 * ece
    return 100.0 * (1.0 - float(np.clip(composite, 0.0, 1.0)))


def _failure_score(fail_data: dict) -> float:
    """
    Compute failure sub-score (0–100).

    FailScore = 100 × clip(confidence_gap, 0, 1)

    A large gap means the model is confident when right and uncertain when
    wrong — the ideal behaviour.
    """
    gap_data = fail_data.get("confidence_gap", {})
    gap = float(gap_data.get("gap", 0.0))

    # Also penalize high-confidence misclassifications
    misc = fail_data.get("misclassification_summary", {})
    overall = misc.get("__overall__", {})
    error_rate = float(overall.get("overall_error_rate", 0.5))

    # Combine: gap contribution (80%) + accuracy contribution (20%)
    gap_score = float(np.clip(gap, 0.0, 1.0))
    acc_score = 1.0 - float(np.clip(error_rate, 0.0, 1.0))
    score = 0.8 * gap_score + 0.2 * acc_score
    return 100.0 * float(np.clip(score, 0.0, 1.0))


def _bias_score(bias_data: dict) -> float:
    """
    Compute bias sub-score (0–100).

    BiasScore = 100 × (1 − clip(bias_penalty, 0, 1))
    bias_penalty = 0.5 × clip(imbalance_ratio/20, 0, 1)
           + 0.5 × max_subgroup_performance_gap
    """
    imbalance = bias_data.get("class_imbalance", {})
    ratio = float(imbalance.get("imbalance_ratio", 1.0))
    imbalance_penalty = float(np.clip((ratio - 1.0) / 19.0, 0.0, 1.0))

    # Subgroup performance gap (worst across all sensitive features)
    max_gap = 0.0
    subgroup = bias_data.get("subgroup_performance", {})
    for feat_data in subgroup.values():
        summary = feat_data.get("__summary__", {})
        gap = float(summary.get("performance_gap", 0.0))
        max_gap = max(max_gap, gap)

    subgroup_penalty = float(np.clip(max_gap, 0.0, 1.0))

    bias_penalty = 0.5 * imbalance_penalty + 0.5 * subgroup_penalty
    return 100.0 * (1.0 - float(np.clip(bias_penalty, 0.0, 1.0)))


def _representation_score(rep_data: dict) -> float:
    """
    Compute representation sub-score (0–100).

    RepScore = 100 × clip(0.5 + 0.5 × silhouette, 0, 1)
    """
    sep = rep_data.get("separability", {})
    sil = float(sep.get("silhouette_score", 0.0))
    if np.isnan(sil):
        sil = 0.0
    return 100.0 * float(np.clip(0.5 + 0.5 * sil, 0.0, 1.0))


# ---------------------------------------------------------------------------
# TrustScoreResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrustScoreResult:
    """
    Structured result from the Trust Score computation.

    Attributes
    ----------
    score : int
      Overall Trust Score in [0, 100].
    grade : str
      Letter grade: A / B / C / D.
    verdict : str
      Plain-English deployment recommendation.
    sub_scores : dict
      Per-dimension scores in [0, 100].
    weights_used : dict
      Actual weights used (after redistribution for missing dimensions).
    breakdown : dict
      Weighted contribution of each dimension to the final score.
    """

    score: int
    grade: str
    verdict: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Trust Score: {self.score}/100 [{self.grade}]",
            f"Assessment : {self.verdict}",
            "\nDimension Breakdown:",
        ]
        for dim, score in self.sub_scores.items():
            lines.append(f"  - {dim:<18} {score:5.1f}/100")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TrustScoreResult(score={self.score}, grade={self.grade!r})"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        from trustlens.visualization.summary_plot import _C, _color_for_grade, _color_for_score

        gc = _color_for_grade(self.grade)

        html = f"""
        <div style="font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    max-width: 450px; padding: 20px; border-radius: 12px;
                    border: 1px solid {gc}40; background-color: #ffffff;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 10px 0;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
                <div style="font-size: 14px; font-weight: 600; color: {_C["gray"]}; text-transform: uppercase; letter-spacing: 0.5px;">
                    Trust Analysis Result
                </div>
                <div style="padding: 4px 12px; border-radius: 20px; background-color: {gc}; color: white;
                            font-size: 13px; font-weight: 700;">
                    GRADE {self.grade}
                </div>
            </div>

            <div style="display: flex; align-items: baseline; margin-bottom: 8px;">
                <span style="font-size: 48px; font-weight: 800; color: {gc}; line-height: 1;">{self.score}</span>
                <span style="font-size: 20px; font-weight: 600; color: {_C["gray"]}; margin-left: 4px;">/100</span>
            </div>

            <div style="font-size: 16px; font-weight: 600; color: {_C["dark"]}; margin-bottom: 20px;">
                {self.verdict}
            </div>

            <div style="border-top: 1px solid #f0f0f0; pt: 15px;">
                <div style="font-size: 12px; font-weight: 700; color: {_C["gray"]}; margin: 12px 0 8px 0; text-transform: uppercase;">
                    Dimension Breakdown
                </div>
        """

        for dim, score in self.sub_scores.items():
            sc = _color_for_score(score)
            html += f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                        <span style="color: {_C["dark"]}; font-weight: 500;">{dim.capitalize()}</span>
                        <span style="color: {sc}; font-weight: 700;">{score:.1f}</span>
                    </div>
                    <div style="width: 100%; height: 6px; background-color: #f0f0f0; border-radius: 3px; overflow: hidden;">
                        <div style="width: {score}%; height: 100%; background-color: {sc}; border-radius: 3px;"></div>
                    </div>
                </div>
            """

        html += """
            </div>
        </div>
        """
        return html


def _score_bar(score: float, width: int = 12) -> str:
    """Return empty string (ASCII bars removed for professional output)."""
    return ""


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------


def compute_trust_score(
    results: dict,
    weights: dict[str, float] | None = None,
) -> TrustScoreResult:
    """
    Compute the overall Trust Score from a TrustReport's results dict.

    Parameters
    ----------
    results : dict
      The ``TrustReport.results`` dictionary.
    weights : dict, optional
      Custom dimension weights. Keys: ``"calibration"``, ``"failure"``,
      ``"bias"``, ``"representation"``. Values must sum to 1.0.
      If None, uses default weights.

    Returns
    -------
    TrustScoreResult
      Structured score result with per-dimension breakdown.

    Examples
    --------
    >>> from trustlens.trust_score import compute_trust_score
    >>> result = compute_trust_score(report.results)
    >>> print(result)
    >>> print(result.score)  # e.g. 74
    >>> print(result.grade)  # e.g. 'B'
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # ------------------------------------------------------------------
    # 1. Compute available sub-scores
    # ------------------------------------------------------------------
    sub_scores: dict[str, float] = {}

    if "calibration" in results:
        sub_scores["calibration"] = _calibration_score(results["calibration"])

    if "failure" in results:
        sub_scores["failure"] = _failure_score(results["failure"])

    if "bias" in results:
        sub_scores["bias"] = _bias_score(results["bias"])

    if "representation" in results:
        sub_scores["representation"] = _representation_score(results["representation"])

    # ------------------------------------------------------------------
    # 2. Redistribute weights for missing dimensions
    # ------------------------------------------------------------------
    active_dims = [d for d in w if d in sub_scores]
    total_active_weight = sum(w[d] for d in active_dims)

    weights_used: dict[str, float] = {}
    if total_active_weight > 0:
        for dim in active_dims:
            weights_used[dim] = w[dim] / total_active_weight
    else:
        # Fallback: equal weights
        for dim in active_dims:
            weights_used[dim] = 1.0 / len(active_dims) if active_dims else 0.0

    # ------------------------------------------------------------------
    # 3. Weighted sum → final score
    # ------------------------------------------------------------------
    raw_score = sum(sub_scores[d] * weights_used[d] for d in active_dims)
    final_score = int(round(float(np.clip(raw_score, 0.0, 100.0))))

    breakdown = {d: round(sub_scores[d] * weights_used[d], 2) for d in active_dims}

    # ------------------------------------------------------------------
    # 4. Assign grade
    # ------------------------------------------------------------------
    grade, verdict = "D", "Low Trust - serious issues"
    for threshold, g, v in _GRADE_THRESHOLDS:
        if final_score >= threshold:
            grade, verdict = g, v
            break

    return TrustScoreResult(
        score=final_score,
        grade=grade,
        verdict=verdict,
        sub_scores={d: round(sub_scores[d], 1) for d in active_dims},
        weights_used={d: round(weights_used[d], 3) for d in active_dims},
        breakdown=breakdown,
    )
