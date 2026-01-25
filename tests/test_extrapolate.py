"""Unit tests for extrapolation models."""

from unittest.mock import patch

import numpy as np
import pytest
from dataless.exceptions import OptimizationWarning, TrainingDataError
from dataless.extrapolate import (
    AbstractExtrapolation,
    ExpDecayExtrapolation,
    FLExtrapolation,
    PolynomialExtrapolation,
    PYPExtrapolation,
)
from hypothesis import given, settings
from numpy.testing import assert_array_equal
from scipy.optimize import OptimizeResult

from conftest import assert_valid_predictions, sample_size_arrays, training_data_strategy

ALL_MODELS = [PYPExtrapolation, FLExtrapolation, ExpDecayExtrapolation, PolynomialExtrapolation]
STATISTICAL_MODELS = [PYPExtrapolation, FLExtrapolation]  # Models that support both metrics
MONOTONIC_MODELS = [PYPExtrapolation, FLExtrapolation, ExpDecayExtrapolation]

# Standard training data as numpy arrays
TRAINING_N = np.array([10, 100, 1000])
TRAINING_CORRECTNESS = np.array([0.8, 0.5, 0.3])
TRAINING_UNIQUENESS = np.array([0.6, 0.3, 0.15])


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestExtrapolationProperties:
    """Property-based tests for extrapolation models."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    @given(test_sizes=sample_size_arrays)
    @settings(max_examples=20)
    def test_predictions_bounded(self, ModelClass, test_sizes):
        """All model predictions are in [0, 1]."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        predictions = model.predict(test_sizes)
        assert_valid_predictions(predictions, monotonic=False)

    @pytest.mark.parametrize("ModelClass", MONOTONIC_MODELS)
    @given(test_sizes=sample_size_arrays)
    @settings(max_examples=20)
    def test_predictions_monotonically_decreasing(self, ModelClass, test_sizes):
        """Monotonic models have decreasing predictions with sample size."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        predictions = model.predict(np.sort(test_sizes))
        assert_valid_predictions(predictions, monotonic=True)

    @pytest.mark.parametrize("ModelClass", STATISTICAL_MODELS)
    @given(training=training_data_strategy())
    @settings(max_examples=10)
    def test_accepts_valid_training_data(self, ModelClass, training):
        """Models accept valid training data."""
        n, values = training
        model = ModelClass(n, correctness=values)
        assert hasattr(model, "h")


# =============================================================================
# Parametrized Unit Tests
# =============================================================================


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
class TestAllExtrapolations:
    """Parametrized tests for all extrapolation models."""

    def test_training_data_required(self, ModelClass):
        """Training data is required."""
        with pytest.raises(TypeError):
            ModelClass()

    def test_must_provide_metric(self, ModelClass):
        """Must provide either correctness or uniqueness."""
        with pytest.raises(TrainingDataError, match="Must provide"):
            ModelClass(TRAINING_N)

    def test_insufficient_training_data(self, ModelClass):
        """Insufficient training data raises exception."""
        with pytest.raises(TrainingDataError):
            ModelClass(np.array([10, 100]), correctness=np.array([0.8, 0.5]))

    def test_initialization(self, ModelClass):
        """Model initializes and trains automatically."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        # All models should have trained parameters
        assert any(hasattr(model, attr) for attr in ["h", "a", "gamma"])

    def test_prediction_shape(self, ModelClass):
        """Predictions match input shape."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        scalar_pred = model.predict(100)
        assert np.isscalar(scalar_pred) or len(scalar_pred) == 1
        array_pred = model.predict(np.array([100, 200, 300]))
        assert len(array_pred) == 3

    def test_prediction_values(self, ModelClass):
        """Predictions are bounded and finite."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        predictions = model.predict(np.array([50, 500, 5000]))
        assert np.all(np.isfinite(predictions))
        assert_valid_predictions(predictions, monotonic=False)


# =============================================================================
# Abstract Base Class Tests
# =============================================================================


class MockExtrapolation(AbstractExtrapolation):
    """Mock implementation for testing AbstractExtrapolation."""

    INIT_STATE = (0.0,)

    def __init__(self, n, *, correctness=None, uniqueness=None):
        self.n_training = np.asarray(n)
        self.values_training = np.asarray(correctness if correctness is not None else uniqueness)
        self.metric = "correctness" if correctness is not None else "uniqueness"
        self.trained = False

    def _make_loss_fn(self):
        return lambda x: 0.0

    def train(self):
        self.trained = True

    def predict_correctness(self, n):
        return np.ones_like(np.atleast_1d(n), dtype=float)

    def predict_uniqueness(self, n):
        return np.ones_like(np.atleast_1d(n), dtype=float) * 0.5

    def _get_param_str(self):
        return "mock=1.0"


class TestAbstractExtrapolation:
    """Tests for AbstractExtrapolation base class."""

    def test_abstract_methods(self):
        """Abstract methods must be implemented."""
        with pytest.raises(TypeError):
            AbstractExtrapolation()

    def test_mock_implementation(self):
        """Concrete implementation works."""
        model = MockExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        assert not model.trained
        model.train()
        assert model.trained
        assert_array_equal(model.predict(np.array([1, 10, 100])), np.ones(3))

    def test_validation_length_mismatch(self):
        """Validation rejects mismatched array lengths."""
        model = MockExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        with pytest.raises(TrainingDataError, match="same length"):
            model.validate_training_data(np.array([1, 2, 3]), np.array([0.5, 0.5]), "correctness")

    def test_validation_insufficient_samples(self):
        """Validation rejects insufficient samples."""
        model = MockExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        with pytest.raises(TrainingDataError, match="at least 3"):
            model.validate_training_data(np.array([1, 2]), np.array([0.5, 0.5]), "correctness")

    def test_validation_invalid_values(self):
        """Validation rejects values outside [0, 1]."""
        model = MockExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        with pytest.raises(TrainingDataError, match="between 0 and 1"):
            model.validate_training_data(np.array([1, 2, 3]), np.array([0.5, 1.5, 0.5]), "correctness")

    def test_validation_negative_n(self):
        """Validation rejects non-positive n values."""
        model = MockExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        with pytest.raises(TrainingDataError, match="positive"):
            model.validate_training_data(np.array([-1, 2, 3]), np.array([0.5, 0.5, 0.5]), "correctness")


# =============================================================================
# Model-Specific Tests
# =============================================================================


class TestExpDecayExtrapolation:
    """Tests specific to ExpDecayExtrapolation."""

    def test_exp_decay_function(self):
        """Static exp decay function works correctly."""
        n = np.array([1, 10, 100])
        result = ExpDecayExtrapolation._compute_static((1.0, 0.1), n)
        assert np.all(np.isfinite(result))
        assert np.all(np.diff(result) <= 0)

    def test_train_on_correctness(self):
        """Can train on correctness and predict correctness."""
        model = ExpDecayExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        assert model.metric == "correctness"
        result = model.predict_correctness(5000)
        assert 0 <= result <= 1

    def test_train_on_uniqueness(self):
        """Can train on uniqueness and predict uniqueness."""
        model = ExpDecayExtrapolation(TRAINING_N, uniqueness=TRAINING_UNIQUENESS)
        assert model.metric == "uniqueness"
        result = model.predict_uniqueness(5000)
        assert 0 <= result <= 1

    def test_predict_wrong_metric_raises(self):
        """Predicting wrong metric raises NotImplementedError."""
        model = ExpDecayExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        with pytest.raises(NotImplementedError, match="trained on correctness"):
            model.predict_uniqueness(100)

        model2 = ExpDecayExtrapolation(TRAINING_N, uniqueness=TRAINING_UNIQUENESS)
        with pytest.raises(NotImplementedError, match="trained on uniqueness"):
            model2.predict_correctness(100)

    def test_predict_returns_trained_metric(self):
        """predict() returns the metric the model was trained on."""
        model = ExpDecayExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        assert np.allclose(model.predict(5000), model.predict_correctness(5000))

        model2 = ExpDecayExtrapolation(TRAINING_N, uniqueness=TRAINING_UNIQUENESS)
        assert np.allclose(model2.predict(5000), model2.predict_uniqueness(5000))


class TestPolynomialExtrapolation:
    """Tests specific to PolynomialExtrapolation."""

    def test_polynomial_function(self):
        """Static polynomial function works correctly."""
        n = np.array([1, 10, 100])
        result = PolynomialExtrapolation._compute_static((-0.1, -0.2, -0.3), n)
        assert np.all(np.isfinite(result))

    def test_train_on_correctness(self):
        """Can train on correctness and predict correctness."""
        model = PolynomialExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        assert model.metric == "correctness"
        result = model.predict_correctness(5000)
        assert 0 <= result <= 1

    def test_train_on_uniqueness(self):
        """Can train on uniqueness and predict uniqueness."""
        model = PolynomialExtrapolation(TRAINING_N, uniqueness=TRAINING_UNIQUENESS)
        assert model.metric == "uniqueness"
        result = model.predict_uniqueness(5000)
        assert 0 <= result <= 1

    def test_predict_wrong_metric_raises(self):
        """Predicting wrong metric raises NotImplementedError."""
        model = PolynomialExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        with pytest.raises(NotImplementedError, match="trained on correctness"):
            model.predict_uniqueness(100)

        model2 = PolynomialExtrapolation(TRAINING_N, uniqueness=TRAINING_UNIQUENESS)
        with pytest.raises(NotImplementedError, match="trained on uniqueness"):
            model2.predict_correctness(100)

    def test_predict_returns_trained_metric(self):
        """predict() returns the metric the model was trained on."""
        model = PolynomialExtrapolation(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        assert np.allclose(model.predict(5000), model.predict_correctness(5000))

        model2 = PolynomialExtrapolation(TRAINING_N, uniqueness=TRAINING_UNIQUENESS)
        assert np.allclose(model2.predict(5000), model2.predict_uniqueness(5000))


# =============================================================================
# Statistical Model Tests (PYP and FL)
# =============================================================================


@pytest.mark.parametrize("ModelClass", STATISTICAL_MODELS)
class TestStatisticalModels:
    """Tests for models that support both correctness and uniqueness."""

    def test_train_on_correctness(self, ModelClass):
        """Can train on correctness data."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        assert model.metric == "correctness"
        assert hasattr(model, "h")

    def test_train_on_uniqueness(self, ModelClass):
        """Can train on uniqueness data."""
        model = ModelClass(TRAINING_N, uniqueness=TRAINING_UNIQUENESS)
        assert model.metric == "uniqueness"
        assert hasattr(model, "h")

    def test_cannot_provide_both_metrics(self, ModelClass):
        """Cannot provide both correctness and uniqueness."""
        with pytest.raises(TrainingDataError, match="not both"):
            ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS, uniqueness=TRAINING_UNIQUENESS)

    def test_predict_correctness(self, ModelClass):
        """predict_correctness returns bounded values."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        result = model.predict_correctness(np.array([50, 500, 5000]))
        assert np.all((result >= 0) & (result <= 1))

    def test_predict_uniqueness(self, ModelClass):
        """predict_uniqueness returns bounded values."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        result = model.predict_uniqueness(np.array([50, 500, 5000]))
        assert np.all((result >= 0) & (result <= 1))

    def test_predict_alias(self, ModelClass):
        """predict() is alias for predict_correctness()."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        n = np.array([50, 500, 5000])
        assert np.allclose(model.predict(n), model.predict_correctness(n))


# =============================================================================
# Deprecation and Summary Tests
# =============================================================================


class TestDeprecationWarning:
    """Tests for deprecation warning on test() method."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_test_method_warns(self, ModelClass):
        """Using test() method emits a deprecation warning."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        with pytest.warns(DeprecationWarning, match="predict"):
            model.test(np.array([100]))

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_test_and_predict_same_output(self, ModelClass):
        """test() and predict() return the same values."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        n = np.array([50, 500, 5000])
        with pytest.warns(DeprecationWarning):
            test_result = model.test(n)
        predict_result = model.predict(n)
        assert np.allclose(test_result, predict_result)


class TestSummaryMethod:
    """Tests for summary() method."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_summary_returns_string(self, ModelClass):
        """summary() returns a non-empty string."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        summary = model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_summary_contains_training_info(self, ModelClass):
        """summary() contains training data info."""
        model = ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
        summary = model.summary()
        assert "Training points: 3" in summary
        assert "10" in summary  # n_min
        assert "1,000" in summary or "1000" in summary  # n_max


# =============================================================================
# Optimization Warning Tests
# =============================================================================


class TestOptimizationWarning:
    """Tests for optimization convergence warnings."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_warns_on_non_convergence(self, ModelClass):
        """Non-convergence emits OptimizationWarning."""
        # Create a mock result that indicates non-convergence
        # Need different x shapes for different models
        if ModelClass == PYPExtrapolation:
            x = [12.0, 0.26]
        elif ModelClass == FLExtrapolation:
            x = [12.0]
        elif ModelClass == ExpDecayExtrapolation:
            x = [1.0, 1.0]
        else:  # PolynomialExtrapolation
            x = [0.0, -1.0, -1.0]

        mock_result = OptimizeResult(x=x, success=False, message="Test non-convergence")

        with patch("dataless.extrapolate.minimize", return_value=mock_result):
            with pytest.warns(OptimizationWarning, match="did not converge"):
                ModelClass(TRAINING_N, correctness=TRAINING_CORRECTNESS)
