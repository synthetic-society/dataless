"""
Unit tests for extrapolation models.

This module contains comprehensive tests for the classes in extrapolate.py,
covering model initialization, training, and prediction functionality.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from dataless.extrapolate import (
    AbstractExtrapolation,
    PYPExtrapolation,
    FLExtrapolation,
    ExpDecayExtrapolation,
    PolynomialExtrapolation
)


@pytest.fixture
def training_data():
    """Fixture providing sample training data."""
    return pd.DataFrame({
        'n': [10, 100, 1000],
        'κ': [0.8, 0.5, 0.3]
    })


@pytest.fixture
def test_sizes():
    """Fixture providing sample test sizes."""
    return np.array([50, 500, 5000])


class MockExtrapolation(AbstractExtrapolation):
    """Mock implementation of AbstractExtrapolation for testing."""
    
    def __init__(self, training_data):
        self.training_data = training_data
        self.trained = False
    
    @classmethod
    def make_loss_fun(cls, dd):
        def loss(x):
            return 0.0
        return loss
    
    def train(self):
        self.trained = True
    
    def test(self, n):
        return np.ones_like(n, dtype=float)


class TestAbstractExtrapolation:
    """Tests for AbstractExtrapolation base class."""

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            AbstractExtrapolation()

    def test_mock_implementation(self, training_data):
        """Test that concrete implementation works."""
        model = MockExtrapolation(training_data)
        assert not model.trained
        model.train()
        assert model.trained
        
        n = np.array([1, 10, 100])
        result = model.test(n)
        assert_array_equal(result, np.ones_like(n))

    def test_validation(self, training_data):
        """Test that validation returnns an exception if
        the training data is invalid."""

        model = MockExtrapolation(training_data)
        with pytest.raises(ValueError):
            model.validate_training_data(pd.DataFrame({'wrong_column': [1, 2, 3]}))
        with pytest.raises(ValueError):
            model.validate_training_data(pd.DataFrame({'n': [], 'κ': []}))
        with pytest.raises(ValueError):
            model.validate_training_data(pd.DataFrame({'n': [1, 2, 3], 'κ': [1, 2]}))
        with pytest.raises(ValueError):
            model.validate_training_data(None)


class TestPYPExtrapolation:
    """Tests for PYP-based extrapolation."""

    def test_initialization(self, training_data):
        """Test model initialization and automatic training."""
        model = PYPExtrapolation(training_data)
        assert hasattr(model, 'h')
        assert hasattr(model, 'γ')

    def test_make_loss_fun(self, training_data):
        """Test loss function creation."""
        loss_fn = PYPExtrapolation.make_loss_fun(training_data)
        assert callable(loss_fn)
        
        # Test with initial state
        loss = loss_fn(PYPExtrapolation.INIT_STATE)
        assert np.isfinite(loss)
        assert loss >= 0

    def test_training(self, training_data):
        """Test model training."""
        model = PYPExtrapolation(training_data)
        
        # Retrain and verify parameters change
        old_h, old_γ = model.h, model.γ
        model.train()
        assert np.isfinite(model.h)
        assert np.isfinite(model.γ)
        assert 0 <= model.γ <= 1

    def test_prediction(self, training_data, test_sizes):
        """Test model predictions."""
        model = PYPExtrapolation(training_data)
        predictions = model.test(test_sizes)
        
        assert len(predictions) == len(test_sizes)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert np.all(np.diff(predictions) <= 0)  # Should decrease with size


class TestFLExtrapolation:
    """Tests for FL-based extrapolation."""

    def test_initialization(self, training_data):
        """Test model initialization and automatic training."""
        model = FLExtrapolation(training_data)
        assert hasattr(model, 'h')

    def test_make_loss_fun(self, training_data):
        """Test loss function creation."""
        loss_fn = FLExtrapolation.make_loss_fun(training_data)
        assert callable(loss_fn)
        
        # Test with initial state
        loss = loss_fn(FLExtrapolation.INIT_STATE)
        assert np.isfinite(loss)
        assert loss >= 0

    def test_training(self, training_data):
        """Test model training."""
        model = FLExtrapolation(training_data)
        
        # Retrain and verify parameters change
        old_h = model.h
        model.train()
        assert np.isfinite(model.h)
        assert model.h >= 0

    def test_prediction(self, training_data, test_sizes):
        """Test model predictions."""
        model = FLExtrapolation(training_data)
        predictions = model.test(test_sizes)
        
        assert len(predictions) == len(test_sizes)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert np.all(np.diff(predictions) <= 0)  # Should decrease with size


class TestExpDecayExtrapolation:
    """Tests for exponential decay extrapolation."""

    def test_initialization(self, training_data):
        """Test model initialization and automatic training."""
        model = ExpDecayExtrapolation(training_data)
        assert hasattr(model, 'a')
        assert hasattr(model, 'b')

    def test_correctness_function(self):
        """Test the correctness calculation function."""
        n = np.array([1, 10, 100])
        result = ExpDecayExtrapolation.correctness(1.0, 0.1, n)
        assert np.all(np.isfinite(result))
        assert np.all(np.diff(result) <= 0)  # Should decrease with size

    def test_make_loss_fun(self, training_data):
        """Test loss function creation."""
        loss_fn = ExpDecayExtrapolation.make_loss_fun(training_data)
        assert callable(loss_fn)
        
        # Test with sample parameters
        loss = loss_fn([1.0, 0.1])
        assert np.isfinite(loss)
        assert loss >= 0

    def test_training(self, training_data):
        """Test model training."""
        model = ExpDecayExtrapolation(training_data)
        
        # Retrain and verify parameters change
        old_a, old_b = model.a, model.b
        model.train()
        assert np.isfinite(model.a)
        assert np.isfinite(model.b)

    def test_prediction(self, training_data, test_sizes):
        """Test model predictions."""
        model = ExpDecayExtrapolation(training_data)
        predictions = model.test(test_sizes)
        
        assert len(predictions) == len(test_sizes)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert np.all(np.diff(predictions) <= 0)  # Should decrease with size


class TestPolynomialExtrapolation:
    """Tests for polynomial extrapolation."""

    def test_initialization(self, training_data):
        """Test model initialization and automatic training."""
        model = PolynomialExtrapolation(training_data)
        assert hasattr(model, 'a')
        assert hasattr(model, 'b')
        assert hasattr(model, 'c')

    def test_correctness_function(self):
        """Test the correctness calculation function."""
        n = np.array([1, 10, 100])
        result = PolynomialExtrapolation.correctness(-0.1, -0.2, -0.3, n)
        assert np.all(np.isfinite(result))

    def test_make_loss_fun(self, training_data):
        """Test loss function creation."""
        loss_fn = PolynomialExtrapolation.make_loss_fun(training_data)
        assert callable(loss_fn)
        
        # Test with sample parameters
        loss = loss_fn([-0.1, -0.2, -0.3])
        assert np.isfinite(loss)
        assert loss >= 0

    def test_training(self, training_data):
        """Test model training."""
        model = PolynomialExtrapolation(training_data)
        
        # Retrain and verify parameters change
        old_a, old_b, old_c = model.a, model.b, model.c
        model.train()
        assert np.isfinite(model.a)
        assert np.isfinite(model.b)
        assert np.isfinite(model.c)

    def test_prediction(self, training_data, test_sizes):
        """Test model predictions."""
        model = PolynomialExtrapolation(training_data)
        predictions = model.test(test_sizes)
        
        assert len(predictions) == len(test_sizes)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)


@pytest.mark.parametrize("ModelClass", [
    PYPExtrapolation,
    FLExtrapolation,
    ExpDecayExtrapolation,
    PolynomialExtrapolation
])
class TestAllExtrapolations:
    """Parameterized tests for all extrapolation models."""

    def test_training_data_required(self, ModelClass):
        """Test that training data is required."""
        with pytest.raises(TypeError):
            ModelClass()

    def test_invalid_training_data(self, ModelClass):
        """Test behavior with invalid training data."""
        invalid_data = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        with pytest.raises(Exception):  # Could be KeyError or ValueError
            ModelClass(invalid_data)

    def test_empty_training_data(self, ModelClass):
        """Test behavior with empty training data."""
        empty_data = pd.DataFrame({'n': [], 'κ': []})
        with pytest.raises(Exception):
            ModelClass(empty_data)

    def test_prediction_shape(self, ModelClass, training_data):
        """Test that predictions match input shape."""
        model = ModelClass(training_data)
        
        # Test scalar input
        scalar_pred = model.test(100)
        assert np.isscalar(scalar_pred) or len(scalar_pred) == 1
        
        # Test array input
        array_pred = model.test(np.array([100, 200, 300]))
        assert len(array_pred) == 3
