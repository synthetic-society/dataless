"""
Unit tests for statistical models.

This module contains comprehensive tests for the functions and classes in model.py,
covering core PYP functionality, utility functions, and model implementations.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from dataless.model import (
    pyp_entropy,
    pyp_uniqueness,
    pyp_correctness,
    invdigamma,
    multiplicities_from_sample,
    multiplicities_from_freqs,
    freqs_from_multiplicities,
    PYP,
    FLModel
)


@pytest.fixture
def valid_pyp_params():
    """Fixture providing valid PYP parameters."""
    return {
        'd': 0.5,
        'α': 1.0
    }


@pytest.fixture
def valid_sample():
    """Fixture providing a valid sample for testing."""
    return np.array([1, 1, 2, 2, 2, 3, 4])


class TestPYPFunctions:
    """Tests for core PYP functions."""

    def test_pyp_entropy_basic(self, valid_pyp_params):
        """Test basic entropy calculation."""
        result = pyp_entropy(valid_pyp_params['d'], valid_pyp_params['α'])
        assert result > 0
        assert np.isfinite(result)

    def test_pyp_entropy_zero_discount(self):
        """Test entropy with zero discount parameter."""
        result = pyp_entropy(0.0, 1.0)
        assert result > 0
        assert np.isfinite(result)

    def test_pyp_uniqueness_basic(self, valid_pyp_params):
        """Test basic uniqueness calculation."""
        n = np.array([1, 10, 100])
        result = pyp_uniqueness(valid_pyp_params['d'], valid_pyp_params['α'], n)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert np.all(np.diff(result) <= 0)  # Should be monotonically decreasing

    def test_pyp_uniqueness_single_sample(self, valid_pyp_params):
        """Test uniqueness for single sample."""
        result = pyp_uniqueness(valid_pyp_params['d'], valid_pyp_params['α'], 1)
        assert_allclose(result, 1.0)

    def test_pyp_correctness_basic(self, valid_pyp_params):
        """Test basic correctness calculation."""
        n = np.array([1, 10, 100])
        result = pyp_correctness(valid_pyp_params['d'], valid_pyp_params['α'], n)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert np.all(np.diff(result) <= 0)  # Should be monotonically decreasing

    def test_pyp_correctness_single_sample(self, valid_pyp_params):
        """Test correctness for single sample."""
        result = pyp_correctness(valid_pyp_params['d'], valid_pyp_params['α'], 1)
        assert_allclose(result, 1.0)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_invdigamma_basic(self):
        """Test basic inverse digamma calculation."""
        y = 2.0
        result = invdigamma(y)
        assert result > 0
        assert np.isfinite(result)

    def test_invdigamma_zero(self):
        """Test inverse digamma at zero."""
        result = invdigamma(0.0)
        assert result > 0
        assert np.isfinite(result)

    def test_invdigamma_negative(self):
        """Test inverse digamma with negative input."""
        result = invdigamma(-1.0)
        assert np.isfinite(result)

    def test_multiplicities_from_sample(self, valid_sample):
        """Test multiplicities calculation from sample."""
        mm, icts = multiplicities_from_sample(valid_sample)
        assert len(mm) == len(icts)
        assert np.sum(mm * icts) == len(valid_sample)

    def test_multiplicities_from_freqs(self):
        """Test multiplicities calculation from frequencies."""
        freqs = np.array([1, 1, 2, 3])
        mm, icts = multiplicities_from_freqs(freqs)
        assert len(mm) == len(icts)
        assert np.sum(mm * icts) == np.sum(freqs)

    def test_freqs_from_multiplicities(self):
        """Test frequency reconstruction from multiplicities."""
        mm = np.array([2, 1])
        icts = np.array([1, 2])
        result = freqs_from_multiplicities(mm, icts)
        expected = np.array([1, 1, 2])
        assert_array_equal(result, expected)


class TestPYP:
    """Tests for PYP class."""

    def test_init_with_d_alpha(self, valid_pyp_params):
        """Test PYP initialization with d and α."""
        pyp = PYP(d=valid_pyp_params['d'], α=valid_pyp_params['α'])
        assert pyp.d == valid_pyp_params['d']
        assert pyp.α == valid_pyp_params['α']

    def test_init_with_h_gamma(self):
        """Test PYP initialization with h and γ."""
        h, gamma = 2.0, 0.5
        pyp = PYP(h=h, γ=gamma)
        assert 0 <= pyp.d < 1
        assert pyp.α > -pyp.d

    def test_init_invalid(self):
        """Test PYP initialization with invalid parameters."""
        with pytest.raises(ValueError):
            PYP()  # No parameters
        with pytest.raises(ValueError):
            PYP(d=0.5)  # Missing α
        with pytest.raises(ValueError):
            PYP(h=2.0)  # Missing γ

    def test_properties(self, valid_pyp_params):
        """Test PYP property calculations."""
        pyp = PYP(d=valid_pyp_params['d'], α=valid_pyp_params['α'])
        assert np.isfinite(pyp.h)
        assert 0 <= pyp.γ <= 1

    def test_methods(self, valid_pyp_params):
        """Test PYP method calculations."""
        pyp = PYP(d=valid_pyp_params['d'], α=valid_pyp_params['α'])
        n = np.array([1, 10, 100])
        
        # Test uniqueness
        u = pyp.uniqueness(n)
        assert np.all(u >= 0)
        assert np.all(u <= 1)
        
        # Test correctness
        c = pyp.correctness(n)
        assert np.all(c >= 0)
        assert np.all(c <= 1)
        
        # Test k-anonymity violations
        k = 2
        v = pyp.kanon_violations(n[0], k)
        assert 0 <= v <= 1


class TestFLModel:
    """Tests for FLModel class."""

    def test_init(self):
        """Test FLModel initialization."""
        h = 2.0
        model = FLModel(h)
        assert model.h == h

    def test_init_clipping(self):
        """Test entropy parameter clipping."""
        model = FLModel(-1.0)
        assert model.h == 0.0
        model = FLModel(150.0)
        assert model.h == 100.0

    def test_methods(self):
        """Test FLModel method calculations."""
        model = FLModel(2.0)
        n = np.array([1, 10, 100])
        
        # Test uniqueness
        u = model.uniqueness(n)
        assert np.all(u >= 0)
        assert np.all(u <= 1)
        assert np.all(np.diff(u) <= 0)  # Should be monotonically decreasing
        
        # Test correctness
        c = model.correctness(n)
        assert np.all(c >= 0)
        assert np.all(c <= 1)
        assert np.all(np.diff(c) <= 0)  # Should be monotonically decreasing

    def test_kanon_violations_not_implemented(self):
        """Test k-anonymity violations raises NotImplementedError."""
        model = FLModel(2.0)
        with pytest.raises(NotImplementedError):
            model.kanon_violations(10, 2)


@pytest.mark.parametrize("n,k", [
    (10, 2),
    (100, 5),
    (1000, 10)
])
def test_kanon_violations_ranges(valid_pyp_params, n, k):
    """Test k-anonymity violations for different n,k combinations."""
    pyp = PYP(d=valid_pyp_params['d'], α=valid_pyp_params['α'])
    result = pyp.kanon_violations(n, k)
    assert 0 <= result <= 1
