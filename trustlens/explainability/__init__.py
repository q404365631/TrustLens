"""
trustlens.explainability.
========================
Explainability sub-package.

Provides gradient-based visual attribution methods for deep models,
and faithfulness evaluation tests for any explanation method.

Modules
-------
* ``gradcam``   — Gradient-weighted Class Activation Maps (Grad-CAM)
* ``faithfulness`` — Pixel deletion/insertion tests for faithfulness
"""

from trustlens.explainability.faithfulness import (
    pixel_deletion_test,
    pixel_insertion_test,
)
from trustlens.explainability.gradcam import GradCAM

__all__ = [
    "GradCAM",
    "pixel_deletion_test",
    "pixel_insertion_test",
]
