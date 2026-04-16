# TrustLens — 10 Rewritten GitHub Issues (Engagement-Optimized)

> These are rewrites of issues #1, #7, #16, #17, #19, #20, #22, #25, #36, #44 from the original list.
> Each has been redesigned to maximize contributor engagement.

---

## Issue #1 (Beginner) — Easy Win, Real Impact

### Add Maximum Calibration Error (MCE) — the worst-case calibration metric

**Problem:**
ECE tells you the *average* calibration gap. But what's the *worst bin*?

A model can have excellent ECE but a single catastrophically miscalibrated confidence range.
MCE (Maximum Calibration Error) catches that. It's used in safety-critical deployments where worst-case matters more than average.

**Your task:** Add `maximum_calibration_error(y_true, y_prob, n_bins=10) → float` to `trustlens/metrics/calibration.py`.

**Formula:**
```
MCE = max over all bins of |accuracy(bin) - confidence(bin)|
```

**Expected output:**
```python
from trustlens.metrics.calibration import maximum_calibration_error
mce = maximum_calibration_error(y_true, y_prob)
# e.g. 0.214 (compared to ECE 0.042 — this reveals a hidden danger zone)
```

**Tests required:** At least `test_mce_perfect_is_zero` and `test_mce_geq_ece`.

**Why it matters:** Safety-critical ML (medical, financial) cares about worst-case behavior, not averages. This makes TrustLens useful for regulated industries.

**Difficulty:** Beginner · Est. time: 1–2 hours

---

## Issue #7 (Beginner) — Tiny but Elegant

### Add confidence-correctness Spearman correlation to failure analysis

**Problem:**
The confidence gap tells us the *mean* difference in confidence between correct and incorrect predictions. But what about the *rank correlation*?

A model with high Spearman correlation between confidence and correctness is trustworthy by construction — you can actually use its confidence score to decide when to trust it.

**Your task:** Add `confidence_correctness_correlation(y_true, y_pred, y_prob) → dict` to `trustlens/metrics/failure.py`.

**Expected output:**
```python
result = confidence_correctness_correlation(y_true, y_pred, y_prob)
# {
#  "spearman_r": 0.71,
#  "p_value": 0.000003,
#  "interpretation": "strong positive: confident predictions are more likely correct"
# }
```

**Interpretation guide to implement:**
- `r > 0.7` → strong
- `0.4–0.7` → moderate
- `< 0.4` → weak (model confidence is unreliable)

**Difficulty:** Beginner · Est. time: 2 hours

---

## Issue #16 (Intermediate) — High-Value Explainability Feature

### Implement Eigen-CAM — gradient-free visual explanations for CNNs

**Problem:**
Grad-CAM requires a backward pass. For some model architectures (quantized models, models with non-differentiable ops), gradients are unavailable.

Eigen-CAM is a drop-in replacement — it uses the first principal component of the feature map to generate explanations. No gradients needed. Often more visually coherent.

**Your task:** Implement `EigenCAM` in `trustlens/explainability/eigencam.py`.

**Interface (must match GradCAM):**
```python
from trustlens.explainability import EigenCAM

cam = EigenCAM(model, target_layer=model.layer4[-1])
heatmap = cam.generate(image_tensor)     # shape (H, W), values in [0,1]
fig   = cam.overlay(image_np, heatmap)
```

**Algorithm:**
1. Forward pass → extract feature map `A` of shape `(C, H, W)` from target layer
2. Flatten to `(C, H*W)` → compute first right singular vector via SVD
3. Project: `cam = A.T @ v` → reshape to `(H, W)` → ReLU → normalize

**Reference:** Muhammad & Yeasin (2020), *EigenCAM: Class Activation Map using Principal Components*. IJCNN.

**Tests required:** `test_eigencam_output_shape`, `test_eigencam_values_in_range`, `test_eigencam_no_backward_needed`.

**Difficulty:** Intermediate · Est. time: 4–6 hours

---

## Issue #17 (Intermediate) — Extends Core Metric

### Extend ECE to multi-class classifiers (one-vs-rest label-wise decomposition)

**Problem:**
The current `expected_calibration_error()` only handles binary classifiers. But most real models are multi-class.

Multi-class ECE should compute per-class calibration and return a macro-averaged ECE alongside per-class breakdown.

**Your task:** Extend `expected_calibration_error()` to accept multi-dimensional `y_prob`.

**Expected output:**
```python
# Binary (current behavior, unchanged)
ece = expected_calibration_error(y_true, y_prob_binary) # float

# Multi-class (new behavior)
ece = expected_calibration_error(y_true, y_prob_multiclass)
# {
#  "macro_ece": 0.071,
#  "per_class": {0: 0.045, 1: 0.082, 2: 0.085}
# }
```

**Implementation strategy:** For each class `k`, create binary labels (`y_true == k`) and binary probs (`y_prob[:, k]`), compute ECE, average.

**Backward compatibility required:** If `y_prob` is 1-D, return a float (existing behavior).

**Difficulty:** Intermediate · Est. time: 3–4 hours

---

## Issue #19 (Intermediate) — High DX Impact

### Add HTML report export — `report.to_html()` with embedded charts

**Problem:**
`report.save()` generates PNGs and JSON. But non-technical stakeholders (product managers, regulators, executives) need a single-file shareable report — no Python, no data science setup.

`report.to_html()` should create a self-contained `.html` file with:
- Trust Score prominently displayed with traffic-light color coding
- All 6 summary dashboard panels as embedded base64 PNGs
- Metric tables with conditional formatting (green/amber/red cells)
- No external dependencies (single file, works offline)

**Expected output:**
```python
report.to_html("model_report.html")
# Opens in any browser — share with stakeholders directly
```

**Implementation hint:** Use `matplotlib` to generate PNG bytes → `base64.b64encode` → embed in `<img src="data:image/png;base64,...">`.

**Why it matters:** This is the feature that gets TrustLens into enterprise workflows. A PDF of a model evaluation report. Decision-makers want this.

**Difficulty:** Intermediate · Est. time: 6–8 hours

---

## Issue #20 (Intermediate) — Core UX Feature

### Add `report.critical_failures()` — ranked table of your model's biggest mistakes

**Problem:**
`show_failures()` prints to console. But users want a *programmatic* way to access critical failures for downstream use — logging, dashboards, alerts.

**Your task:** Add `critical_failures(n=20) → list[dict]` to `TrustReport`.

**Expected output:**
```python
failures = report.critical_failures(n=10)
# [
#  {
#   "rank": 1,
#   "sample_index": 412,
#   "y_true": 1,
#   "y_pred": 0,
#   "confidence": 0.974,
#   "danger": "CRITICAL",
#   "features": [0.23, -1.44, 0.87, ...]  # raw feature vector
#  },
#  ...
# ]
```

**Bonus:** Add `report.export_failures(path="failures.csv")` to save as CSV.

**Why it matters:** Enables automated failure monitoring in CI pipelines. If `critical_failures(n=5)` returns items with `confidence > 0.95`, fail the build.

**Difficulty:** Intermediate · Est. time: 2–3 hours

---

## Issue #22 (Intermediate) — Visual WOW

### Add UMAP/t-SNE embedding visualization — see what your model learned

**Problem:**
`embedding_separability()` reports a silhouette score. But a *number* can't show you that your class embeddings overlap almost perfectly in one region, or that there's a clear cluster of your minority class that the model completely ignores.

A 2D scatter plot of embeddings, colored by class, tells that story in one glance.

**Your task:** Add `plot_embedding_2d(embeddings, y_true, method="umap")` to `trustlens/visualization/representation_plots.py`.

**Expected output:**
- 2D scatter plot with per-class color coding
- Silhouette score annotated in the corner
- Optional: show centroids as large markers

**Implementation:**
```python
# Try UMAP first, fall back to t-SNE
try:
  from umap import UMAP
  proj = UMAP(n_components=2).fit_transform(embeddings)
except ImportError:
  from sklearn.manifold import TSNE
  proj = TSNE(n_components=2).fit_transform(embeddings)
```

**UMAP is an optional dependency** — add to `setup.cfg[extras_require][full]`.

**Difficulty:** Intermediate · Est. time: 3–5 hours

---

## Issue #25 (Intermediate) — Fairness Research

### Add Equalized Odds — the fairness metric that catches systematic discrimination

**Problem:**
Subgroup accuracy gap tells you *if* one group is worse off.
Equalized Odds tells you *how* they're worse off — through predictive bias in True Positive Rate or False Positive Rate.

A model can have equal accuracy but a TPR gap of 30% between groups. That's systematic discrimination, invisible to accuracy-based metrics.

**Your task:** Add `equalized_odds(y_true, y_pred, sensitive_features) → dict` to `trustlens/metrics/bias.py`.

**Expected output:**
```python
result = equalized_odds(y_true, y_pred, {"gender": gender_arr})
# {
#  "gender": {
#   "M": {"tpr": 0.88, "fpr": 0.12},
#   "F": {"tpr": 0.61, "fpr": 0.08},
#   "__summary__": {
#    "tpr_gap": 0.27,
#    "fpr_gap": 0.04,
#    "violation": "severe"  # tpr_gap > 0.15
#   }
#  }
# }
```

**Violation thresholds to implement:**
- `> 0.15` → "severe"
- `0.05–0.15` → "moderate"
- `< 0.05` → "acceptable"

**Reference:** Hardt, Price, Srebo (2016), *Equality of Opportunity in Supervised Learning*. NeurIPS.

**Difficulty:** Intermediate · Est. time: 4–5 hours

---

## Issue #36 (Advanced) — The Feature That Gets Conference Talks

### Implement Grad-CAM for Vision Transformers (ViT) — the hardest explainability challenge

**Problem:**
Regular Grad-CAM requires convolutional feature maps. ViTs have attention heads instead.

Explaining ViT predictions is an open research problem. The best current approach uses gradients w.r.t. attention weights in the last transformer block, projected onto the 2D patch grid.

**Your task:** Implement `ViTGradCAM` in `trustlens/explainability/vit_gradcam.py`.

**Interface:**
```python
from trustlens.explainability import ViTGradCAM

cam = ViTGradCAM(vit_model, attention_layer_index=-1)
heatmap = cam.generate(image_tensor, class_idx=None)  # shape (H, W)
fig = cam.overlay(image_np, heatmap)
```

**Algorithm (Chefer et al., 2021):**
1. Run forward pass → extract attention weights from last block
2. Run backward pass → compute gradient of score w.r.t. attention weights
3. Weight attention maps by gradient magnitude
4. Average across heads → reshape from `(num_patches,)` to `(P, P)` patch grid
5. Upsample to input resolution

**Critical constraints:**
- Must handle both HuggingFace `ViTForImageClassification` and `timm` ViT models
- Must gracefully degrade to attention rollout if gradients are unavailable

**Reference:** Chefer, Gur, Wolf (2021), *Transformer Interpretability Beyond Attention Visualization*. CVPR.

**Difficulty:** Advanced · Est. time: 2–3 days

---

## Issue #44 (Advanced) — The Project That Unlocks Enterprise Users

### Build `trustlens serve` — the web dashboard that closes the loop with stakeholders

**Problem:**
TrustLens generates data. Stakeholders want a UI.

A product manager can't read `report.json`. A regulator wants a printable PDF.
A compliance team needs to compare Model v1 vs Model v2.

`trustlens serve report/` should launch a local web dashboard that any stakeholder can use.

**Your task:** Build the TrustLens web dashboard.

**Command:**
```bash
trustlens serve examples/output/ # auto-detects report.json + trust_score.json
# Serving at http://localhost:7788
```

**Tech stack (constraints):**
- **Backend:** FastAPI (already a Python project — keep it consistent)
- **Frontend:** HTMX + minimal vanilla JS — no React, no build step
- **Charts:** Plotly.js via CDN — interactive versions of matplotlib plots
- **No database needed** for single-session use

**Required panels:**
1. Trust Score gauge (interactive — click for dimension breakdown)
2. Reliability diagram (Plotly, interactive hover)
3. Confidence gap histogram
4. Critical failures table (sortable, filterable)
5. Model comparison (if multiple report dirs provided)
6. Export as PDF button (browser print)

**CLI integration:**
```python
# trustlens/cli.py
@app.command()
def serve(report_dir: str, port: int = 7788): ...
```

**Entry point in `setup.cfg`:**
```ini
[options.entry_points]
console_scripts =
  trustlens = trustlens.cli:app
```

**Why it matters:** This is the feature that takes TrustLens from "dev tool" to "team tool." It gets TrustLens into weekly model review meetings. It's the path to 1000+ stars.

**Difficulty:** Advanced · Est. time: 1–2 weeks
