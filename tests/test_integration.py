"""Integration tests for the dataless package."""

import numpy as np
import pytest
from dataless.empirical import correctness, empirical_entropy, frequencies, uniqueness
from dataless.extrapolate import (
    ExpDecayExtrapolation,
    FLExtrapolation,
    PolynomialExtrapolation,
    PYPExtrapolation,
)
from dataless.model import PYP, FLModel, multiplicities_from_freqs
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import assert_bounded, assert_valid_predictions

ALL_MODELS = [PYPExtrapolation, FLExtrapolation, ExpDecayExtrapolation, PolynomialExtrapolation]
MONOTONIC_MODELS = [PYPExtrapolation, FLExtrapolation, ExpDecayExtrapolation]


@pytest.fixture
def synthetic_scaling_data():
    """Synthetic scaling data mimicking real empirical observations."""
    rng = np.random.default_rng(42)
    sample_sizes = [50, 100, 200, 500, 1000]
    kappas = [correctness(frequencies(rng.zipf(a=1.5, size=n))) for n in sample_sizes]
    return (np.array(sample_sizes), np.array(kappas))


# =============================================================================
# Empirical -> Model Pipeline
# =============================================================================


class TestEmpiricalToModel:
    """Integration tests for empirical -> model pipeline."""

    def test_frequencies_to_pyp_model(self):
        """Empirical frequencies can be used with a PYP model."""
        sample = np.random.default_rng(42).zipf(a=2.0, size=1000)
        freqs = frequencies(sample)

        # Empirical statistics are valid
        assert_bounded(correctness(freqs))
        assert_bounded(uniqueness(freqs))
        assert empirical_entropy(sample) >= 0

        # Model predictions are valid
        pyp = PYP(d=0.5, alpha=1.0)
        assert_bounded(pyp.correctness(len(sample)))
        assert_bounded(pyp.uniqueness(len(sample)))

    def test_frequencies_to_flmodel(self):
        """Empirical entropy can initialize an FLModel."""
        sample = np.random.default_rng(42).zipf(a=2.0, size=1000)
        model = FLModel(empirical_entropy(sample, base=2))
        assert_bounded(model.correctness(len(sample)))
        assert_bounded(model.uniqueness(len(sample)))

    def test_multiplicities_preserve_structure(self):
        """Multiplicities correctly represent frequency structure."""
        sample = np.random.default_rng(42).integers(0, 100, size=500)
        freqs = frequencies(sample)
        mm, icts = multiplicities_from_freqs(freqs)
        assert np.sum(mm * icts) == len(sample)
        assert np.sum(mm) == len(freqs)


# =============================================================================
# Empirical -> Extrapolation Pipeline
# =============================================================================


class TestEmpiricalToExtrapolation:
    """Integration tests for empirical -> extrapolation pipeline."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_full_pipeline(self, ModelClass, synthetic_scaling_data):
        """Complete pipeline: empirical data -> extrapolation -> predictions."""
        n, kappas = synthetic_scaling_data
        model = ModelClass(n, correctness=kappas)
        predictions = model.predict(np.array([2000, 5000, 10000]))
        assert np.all(np.isfinite(predictions))
        assert_bounded(predictions)

    @pytest.mark.parametrize("ModelClass", MONOTONIC_MODELS)
    def test_predictions_decreasing(self, ModelClass, synthetic_scaling_data):
        """Monotonic models have decreasing predictions."""
        n, kappas = synthetic_scaling_data
        model = ModelClass(n, correctness=kappas)
        predictions = model.predict(np.array([100, 500, 1000, 5000]))
        assert_valid_predictions(predictions, monotonic=True)

    def test_pyp_fits_training_data(self, synthetic_scaling_data):
        """PYP predictions are close to training values."""
        n, kappas = synthetic_scaling_data
        model = PYPExtrapolation(n, correctness=kappas)
        train_pred = model.predict(n)
        assert np.all(np.abs(train_pred - kappas) < 0.3)


# =============================================================================
# Model Comparison
# =============================================================================


class TestModelComparison:
    """Integration tests comparing different models."""

    COMPARISON_N = np.array([10, 50, 100, 500, 1000])
    COMPARISON_CORRECTNESS = np.array([0.9, 0.7, 0.5, 0.3, 0.2])

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_produces_valid_predictions(self, ModelClass):
        """All models produce valid predictions."""
        predictions = ModelClass(self.COMPARISON_N, correctness=self.COMPARISON_CORRECTNESS).predict(
            np.array([100, 1000, 10000])
        )
        assert np.all(np.isfinite(predictions))
        assert_bounded(predictions)

    @pytest.mark.parametrize("ModelClass", MONOTONIC_MODELS)
    def test_predictions_monotonically_decreasing(self, ModelClass):
        """Monotonic models show decreasing correctness."""
        predictions = ModelClass(self.COMPARISON_N, correctness=self.COMPARISON_CORRECTNESS).predict(
            np.array([100, 500, 1000, 5000])
        )
        assert np.all(np.diff(predictions) <= 1e-6)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Integration tests for edge cases."""

    MINIMAL_N = np.array([10, 100, 1000])
    MINIMAL_CORRECTNESS = np.array([0.9, 0.5, 0.1])

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_minimal_training_data(self, ModelClass):
        """Models handle minimal (3 points) training data."""
        assert np.all(
            np.isfinite(ModelClass(self.MINIMAL_N, correctness=self.MINIMAL_CORRECTNESS).predict(np.array([500])))
        )

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_single_prediction_point(self, ModelClass):
        """Models handle scalar input."""
        pred = ModelClass(self.MINIMAL_N, correctness=self.MINIMAL_CORRECTNESS).predict(500)
        assert np.isscalar(pred) or len(pred) == 1

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_large_extrapolation(self, ModelClass):
        """Models extrapolate far beyond training range."""
        predictions = ModelClass(self.MINIMAL_N, correctness=self.MINIMAL_CORRECTNESS).predict(
            np.array([10000, 100000, 1000000])
        )
        assert np.all(np.isfinite(predictions))
        assert_bounded(predictions)


# =============================================================================
# Property-Based Integration Tests
# =============================================================================


class TestPropertyBasedIntegration:
    """Property-based integration tests."""

    @given(st.lists(st.integers(0, 1000), min_size=10, max_size=500).map(np.array))
    @settings(max_examples=20)
    def test_empirical_statistics_consistency(self, sample):
        """Empirical statistics are internally consistent."""
        freqs = frequencies(sample)
        if len(np.unique(sample)) <= 1:
            return  # Skip trivial cases

        c, u = correctness(freqs), uniqueness(freqs)
        assert u <= c + 1e-10  # Uniqueness <= correctness
        assert_bounded(c)
        assert_bounded(u)
        assert freqs.sum() == len(sample)
