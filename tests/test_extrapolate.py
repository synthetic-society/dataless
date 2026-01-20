"""Unit tests for extrapolation models."""

from unittest.mock import patch

import numpy as np
import pandas as pd
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
MONOTONIC_MODELS = [PYPExtrapolation, FLExtrapolation, ExpDecayExtrapolation]


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestExtrapolationProperties:
    """Property-based tests for extrapolation models."""

    TRAINING = pd.DataFrame({"n": [10, 100, 1000], "κ": [0.8, 0.5, 0.3]})

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    @given(test_sizes=sample_size_arrays)
    @settings(max_examples=20)
    def test_predictions_bounded(self, ModelClass, test_sizes):
        """All model predictions are in [0, 1]."""
        model = ModelClass(self.TRAINING)
        predictions = model.predict(test_sizes)
        assert_valid_predictions(predictions, monotonic=False)

    @pytest.mark.parametrize("ModelClass", MONOTONIC_MODELS)
    @given(test_sizes=sample_size_arrays)
    @settings(max_examples=20)
    def test_predictions_monotonically_decreasing(self, ModelClass, test_sizes):
        """Monotonic models have decreasing predictions with sample size."""
        model = ModelClass(self.TRAINING)
        predictions = model.predict(np.sort(test_sizes))
        assert_valid_predictions(predictions, monotonic=True)

    @pytest.mark.parametrize("ModelClass", [PYPExtrapolation, FLExtrapolation])
    @given(training=training_data_strategy())
    @settings(max_examples=10)
    def test_accepts_valid_training_data(self, ModelClass, training):
        """Models accept valid training data."""
        model = ModelClass(training)
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

    def test_invalid_training_data(self, ModelClass):
        """Invalid training data raises exception."""
        with pytest.raises(TrainingDataError):
            ModelClass(pd.DataFrame({"wrong_column": [1, 2, 3]}))

    def test_empty_training_data(self, ModelClass):
        """Empty training data raises exception."""
        with pytest.raises(TrainingDataError):
            ModelClass(pd.DataFrame({"n": [], "κ": []}))

    def test_initialization(self, ModelClass, training_data):
        """Model initializes and trains automatically."""
        model = ModelClass(training_data)
        # All models should have trained parameters
        assert any(hasattr(model, attr) for attr in ["h", "a", "γ"])

    def test_make_loss_fun(self, ModelClass, training_data):
        """Loss function is callable and returns non-negative finite values."""
        loss_fn = ModelClass.make_loss_fun(training_data)
        assert callable(loss_fn)
        # Use model-specific initial state or default
        init_state = getattr(ModelClass, "INIT_STATE", [1.0] * 3)
        loss = loss_fn(init_state)
        assert np.isfinite(loss) and loss >= 0

    def test_prediction_shape(self, ModelClass, training_data):
        """Predictions match input shape."""
        model = ModelClass(training_data)
        scalar_pred = model.predict(100)
        assert np.isscalar(scalar_pred) or len(scalar_pred) == 1
        array_pred = model.predict(np.array([100, 200, 300]))
        assert len(array_pred) == 3

    def test_prediction_values(self, ModelClass, training_data):
        """Predictions are bounded and finite."""
        model = ModelClass(training_data)
        predictions = model.predict(np.array([50, 500, 5000]))
        assert np.all(np.isfinite(predictions))
        assert_valid_predictions(predictions, monotonic=False)


# =============================================================================
# Abstract Base Class Tests
# =============================================================================


class MockExtrapolation(AbstractExtrapolation):
    """Mock implementation for testing AbstractExtrapolation."""

    INIT_STATE = [0.0]

    def __init__(self, training_data):
        self.training_data = training_data
        self.trained = False

    @classmethod
    def make_loss_fun(cls, dd):
        return lambda x: 0.0

    def train(self):
        self.trained = True

    def predict(self, n):
        return np.ones_like(n, dtype=float)

    def summary(self):
        return "Mock Model"


class TestAbstractExtrapolation:
    """Tests for AbstractExtrapolation base class."""

    def test_abstract_methods(self):
        """Abstract methods must be implemented."""
        with pytest.raises(TypeError):
            AbstractExtrapolation()

    def test_mock_implementation(self, training_data):
        """Concrete implementation works."""
        model = MockExtrapolation(training_data)
        assert not model.trained
        model.train()
        assert model.trained
        assert_array_equal(model.predict(np.array([1, 10, 100])), np.ones(3))

    def test_validation(self, training_data):
        """Validation rejects invalid training data."""
        model = MockExtrapolation(training_data)
        invalid_cases = [
            pd.DataFrame({"wrong_column": [1, 2, 3]}),
            pd.DataFrame({"n": [], "κ": []}),
            None,
        ]
        for invalid in invalid_cases:
            with pytest.raises((TrainingDataError, TypeError)):
                model.validate_training_data(invalid)


# =============================================================================
# Model-Specific Tests
# =============================================================================


class TestExpDecayExtrapolation:
    """Tests specific to ExpDecayExtrapolation."""

    def test_correctness_function(self):
        """Static correctness function works correctly."""
        n = np.array([1, 10, 100])
        result = ExpDecayExtrapolation.correctness(1.0, 0.1, n)
        assert np.all(np.isfinite(result))
        assert np.all(np.diff(result) <= 0)


class TestPolynomialExtrapolation:
    """Tests specific to PolynomialExtrapolation."""

    def test_correctness_function(self):
        """Static correctness function works correctly."""
        n = np.array([1, 10, 100])
        result = PolynomialExtrapolation.correctness(-0.1, -0.2, -0.3, n)
        assert np.all(np.isfinite(result))


# =============================================================================
# Deprecation and Summary Tests
# =============================================================================


class TestDeprecationWarning:
    """Tests for deprecation warning on test() method."""

    TRAINING = pd.DataFrame({"n": [10, 100, 1000], "κ": [0.8, 0.5, 0.3]})

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_test_method_warns(self, ModelClass):
        """Using test() method emits a deprecation warning."""
        model = ModelClass(self.TRAINING)
        with pytest.warns(DeprecationWarning, match="predict"):
            model.test(np.array([100]))

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_test_and_predict_same_output(self, ModelClass):
        """test() and predict() return the same values."""
        model = ModelClass(self.TRAINING)
        n = np.array([50, 500, 5000])
        with pytest.warns(DeprecationWarning):
            test_result = model.test(n)
        predict_result = model.predict(n)
        assert np.allclose(test_result, predict_result)


class TestSummaryMethod:
    """Tests for summary() method."""

    TRAINING = pd.DataFrame({"n": [10, 100, 1000], "κ": [0.8, 0.5, 0.3]})

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_summary_returns_string(self, ModelClass):
        """summary() returns a non-empty string."""
        model = ModelClass(self.TRAINING)
        summary = model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_summary_contains_training_info(self, ModelClass):
        """summary() contains training data info."""
        model = ModelClass(self.TRAINING)
        summary = model.summary()
        assert "Training points: 3" in summary
        assert "10" in summary  # n_min
        assert "1,000" in summary or "1000" in summary  # n_max


# =============================================================================
# Optimization Warning Tests
# =============================================================================


class TestOptimizationWarning:
    """Tests for optimization convergence warnings."""

    TRAINING = pd.DataFrame({"n": [10, 100, 1000], "κ": [0.8, 0.5, 0.3]})

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
                ModelClass(self.TRAINING)
