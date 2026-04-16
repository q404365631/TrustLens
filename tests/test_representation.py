"""
tests/test_representation.py.
=============================
Unit tests for trustlens.metrics.representation.
"""

import numpy as np
import pytest

from trustlens.metrics.representation import (
    centered_kernel_alignment,
    embedding_separability,
)


class TestEmbeddingSeparability:
    def test_basic_output_keys(self):
        embeddings = np.random.randn(50, 10)
        y_true = np.array([0] * 25 + [1] * 25)
        result = embedding_separability(embeddings, y_true)

        expected_keys = {
            "silhouette_score",
            "within_class_distance",
            "between_class_distance",
            "separability_ratio",
            "n_samples_used",
            "embedding_dim",
        }
        assert expected_keys.issubset(result.keys())

    def test_well_separated_has_high_silhouette(self):
        """Classes far apart in embedding space → high silhouette."""
        class_a = np.random.randn(50, 5) + np.array([10, 0, 0, 0, 0])
        class_b = np.random.randn(50, 5) - np.array([10, 0, 0, 0, 0])
        embeddings = np.vstack([class_a, class_b])
        y_true = np.array([0] * 50 + [1] * 50)
        result = embedding_separability(embeddings, y_true)
        assert result["silhouette_score"] > 0.5

    def test_embedding_dim_correct(self):
        embeddings = np.random.randn(30, 16)
        y_true = np.array([0] * 15 + [1] * 15)
        result = embedding_separability(embeddings, y_true)
        assert result["embedding_dim"] == 16

    def test_separability_ratio_positive(self):
        embeddings = np.random.randn(40, 8)
        y_true = np.array([0] * 20 + [1] * 20)
        result = embedding_separability(embeddings, y_true)
        # Ratio can be 'inf' for very tight clusters
        assert result["separability_ratio"] >= 0


class TestCenteredKernelAlignment:
    def test_self_similarity_is_one(self):
        """CKA(X, X) should be exactly 1.0."""
        X = np.random.randn(20, 5)
        cka = centered_kernel_alignment(X, X)
        assert cka == pytest.approx(1.0, rel=1e-5)

    def test_range_zero_to_one(self):
        X = np.random.randn(20, 5)
        Y = np.random.randn(20, 8)
        cka = centered_kernel_alignment(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_shape_mismatch_raises(self):
        X = np.random.randn(20, 5)
        Y = np.random.randn(25, 5)
        with pytest.raises(ValueError, match="same number of samples"):
            centered_kernel_alignment(X, Y)

    def test_orthogonal_representations_low_cka(self):
        """Near-orthogonal representations should have low CKA."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((50, 10))
        # Make Y completely independent of X
        Y = rng.standard_normal((50, 10))
        cka = centered_kernel_alignment(X, Y)
        # We can't guarantee exactly 0, but it should be < 0.9
        assert cka < 0.9
