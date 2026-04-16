# Changelog

All notable changes to TrustLens are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Nothing yet — all current work is in v0.1.0.

---

## [0.1.0] — 2026-04-16

- `trustlens.quick_analyze()` — zero-friction, branded entry point with auto-loading demo data
- `trustlens.analyze()` — primary analysis API with module dispatch
- `TrustReport` result container with rich `_repr_html_` for Jupyter, plus `show()`, `plot()`, `save()`
- **Calibration module**: `brier_score`, `expected_calibration_error`, `reliability_curve`
- **Failure module**: `misclassification_summary`, `confidence_gap`
- **Bias module**: `class_imbalance_report`, `subgroup_performance`
- **Representation module**: `embedding_separability`, `centered_kernel_alignment`
- **Explainability**: `GradCAM` class with hook-based PyTorch implementation
- **Faithfulness**: `pixel_deletion_test`, `pixel_insertion_test` with AUPC metric
- **Visualization**: Professional base64-rendered Jupyter dashboards and premium Matplotlib visualizations
- **UX**: `tqdm` progress tracking for long-running batch analysis
- **Plugin system**: `BasePlugin` ABC + `PluginRegistry` singleton
- Full test suite: `test_calibration`, `test_failure`, `test_bias`, `test_representation`, `test_api`, `test_plugins`
- Examples: `trustlens_demo.ipynb` (Colab-ready), `quickstart.py`, `calibration_deep_dive.py`
- GitHub Actions CI workflow (linting, testing, and formatting)
- Complete documentation: README (with logo), CONTRIBUTING, ROADMAP, this CHANGELOG

[Unreleased]: https://github.com/Khanz9664/TrustLens/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Khanz9664/TrustLens/releases/tag/v0.1.0
