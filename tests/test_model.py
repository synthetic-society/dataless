"""Unit tests for statistical models."""

import numpy as np
import pytest
from dataless.exceptions import ParameterError
from dataless.model import (
    PYP,
    FLModel,
    freqs_from_multiplicities,
    invdigamma,
    multiplicities_from_freqs,
    multiplicities_from_sample,
    pyp_correctness,
    pyp_entropy,
    pyp_uniqueness,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import digamma

from conftest import assert_bounded, entropy_values, frequency_arrays, pyp_params, sample_size_arrays

# =============================================================================
# Property-Based Tests: Inverse Digamma
# =============================================================================


class TestInvdigamma:
    """Tests for the inverse digamma function."""

    @given(st.floats(min_value=-1.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    def test_round_trip(self, y):
        """digamma(invdigamma(y)) ≈ y."""
        x = invdigamma(y)
        assume(np.isfinite(x) and x > 0)
        assert_allclose(digamma(x), y, rtol=1e-4, atol=1e-10)

    @given(st.floats(min_value=0.5, max_value=100.0, allow_nan=False, allow_infinity=False))
    def test_positive_for_positive_input(self, y):
        """Invdigamma returns positive values for positive inputs."""
        assert invdigamma(y) > 0

    @given(st.floats(min_value=-1.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    def test_always_finite(self, y):
        """Invdigamma always returns finite values."""
        assert np.isfinite(invdigamma(y))


# =============================================================================
# Property-Based Tests: Multiplicities
# =============================================================================


class TestMultiplicities:
    """Tests for multiplicities conversion functions."""

    @given(frequency_arrays)
    def test_round_trip_preserves_sorted_freqs(self, freqs):
        """Round-trip through multiplicities preserves sorted frequencies."""
        mm, icts = multiplicities_from_freqs(freqs)
        reconstructed = freqs_from_multiplicities(mm, icts)
        assert_array_equal(np.sort(reconstructed), np.sort(freqs))

    @given(frequency_arrays)
    def test_sum_preserved(self, freqs):
        """sum(mm * icts) == sum(freqs)."""
        mm, icts = multiplicities_from_freqs(freqs)
        assert np.sum(mm * icts) == np.sum(freqs)

    @given(frequency_arrays)
    def test_count_preserved(self, freqs):
        """sum(mm) == len(freqs)."""
        mm, icts = multiplicities_from_freqs(freqs)
        assert np.sum(mm) == len(freqs)

    def test_from_sample(self):
        """multiplicities_from_sample works correctly."""
        sample = np.array([1, 1, 2, 2, 2, 3, 4])
        mm, icts = multiplicities_from_sample(sample)
        assert len(mm) == len(icts)
        assert np.sum(mm * icts) == len(sample)

    def test_freqs_from_multiplicities(self):
        """freqs_from_multiplicities reconstructs frequencies correctly."""
        mm, icts = np.array([2, 1]), np.array([1, 2])
        assert_array_equal(freqs_from_multiplicities(mm, icts), np.array([1, 1, 2]))


# =============================================================================
# Property-Based Tests: PYP Parameters
# =============================================================================


class TestPYPParameters:
    """Tests for PYP parameter conversions and properties."""

    @given(pyp_params())
    def test_h_gamma_round_trip(self, params):
        """PYP(d,alpha) -> (h,gamma) -> PYP(h,gamma) preserves d,alpha."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d)

        pyp1 = PYP(d=d, alpha=alpha)
        h, gamma = pyp1.h, pyp1.gamma
        assume(np.isfinite(h) and np.isfinite(gamma) and 0 < gamma < 1)

        pyp2 = PYP(h=h, gamma=gamma)
        assert_allclose(pyp2.d, d, rtol=1e-3, atol=1e-6)
        assert_allclose(pyp2.alpha, alpha, rtol=1e-3, atol=1e-6)

    @given(pyp_params())
    def test_entropy_positive(self, params):
        """PYP entropy is always positive."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d)
        assert pyp_entropy(d, alpha) > 0

    @given(pyp_params())
    def test_gamma_bounded(self, params):
        """PYP gamma is in (0, 1) for non-trivial d."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d and d >= 0.01)
        assert 0 < PYP(d=d, alpha=alpha).gamma < 1


# =============================================================================
# Property-Based Tests: PYP Functions
# =============================================================================


class TestPYPFunctions:
    """Tests for PYP statistical functions."""

    @given(pyp_params(), sample_size_arrays)
    def test_uniqueness_bounded(self, params, n):
        """PYP uniqueness is always in [0, 1]."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d and d >= 1e-10)
        assert_bounded(pyp_uniqueness(d, alpha, n))

    @given(pyp_params(), sample_size_arrays)
    def test_correctness_bounded(self, params, n):
        """PYP correctness is always in [0, 1]."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d and d >= 1e-6)  # Tighter bound to avoid numerical edge cases
        assert_bounded(pyp_correctness(d, alpha, n), atol=1e-7)

    @given(pyp_params(), sample_size_arrays)
    def test_uniqueness_monotonically_decreasing(self, params, n):
        """PYP uniqueness decreases with sample size."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d)
        u = pyp_uniqueness(d, alpha, np.sort(n))
        assert np.all(np.diff(u) <= 1e-5)

    @given(pyp_params(), sample_size_arrays)
    def test_correctness_monotonically_decreasing(self, params, n):
        """PYP correctness decreases with sample size."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d)
        c = pyp_correctness(d, alpha, np.sort(n))
        assert np.all(np.diff(c) <= 1e-5)

    @given(pyp_params())
    def test_single_sample_correctness_is_one(self, params):
        """Correctness for n=1 is always 1."""
        d, alpha = params["d"], params["alpha"]
        assume(alpha > -d and d >= 1e-10)  # Avoid numerical instability
        assert_allclose(pyp_correctness(d, alpha, 1), 1.0)


# =============================================================================
# Property-Based Tests: FLModel
# =============================================================================


class TestFLModelProperties:
    """Property-based tests for FLModel."""

    @given(entropy_values, sample_size_arrays)
    def test_uniqueness_bounded(self, h, n):
        """FLModel uniqueness is always in [0, 1]."""
        assert_bounded(FLModel(h).uniqueness(n))

    @given(entropy_values, sample_size_arrays)
    def test_correctness_bounded(self, h, n):
        """FLModel correctness is always in [0, 1]."""
        assume(np.all(n >= 1))
        assert_bounded(FLModel(h).correctness(n), high=1.0 + 1e-6)

    @given(entropy_values, sample_size_arrays)
    def test_uniqueness_monotonically_decreasing(self, h, n):
        """FLModel uniqueness decreases with sample size."""
        u = FLModel(h).uniqueness(np.sort(n))
        assert np.all(np.diff(u) <= 1e-10)

    @given(st.floats(min_value=-10.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    def test_entropy_clipping(self, h):
        """FLModel entropy is clipped to [0, 100]."""
        assert 0 <= FLModel(h).h <= 100


# =============================================================================
# Unit Tests: PYP Class
# =============================================================================


class TestPYP:
    """Tests for PYP class."""

    def test_init_with_d_alpha(self):
        """PYP initialization with d and alpha."""
        pyp = PYP(d=0.5, alpha=1.0)
        assert pyp.d == 0.5 and pyp.alpha == 1.0

    def test_init_with_h_gamma(self):
        """PYP initialization with h and gamma."""
        pyp = PYP(h=2.0, gamma=0.5)
        assert 0 <= pyp.d < 1 and pyp.alpha > -pyp.d

    @pytest.mark.parametrize("kwargs", [{}, {"d": 0.5}, {"h": 2.0}])
    def test_init_invalid(self, kwargs):
        """PYP initialization with invalid parameters raises ParameterError."""
        with pytest.raises(ParameterError):
            PYP(**kwargs)

    @pytest.mark.parametrize(
        "d,alpha",
        [
            (1.5, 1.0),  # d >= 1
            (-0.1, 1.0),  # d < 0
            (0.5, -1.0),  # alpha <= -d
        ],
    )
    def test_init_invalid_d_alpha(self, d, alpha):
        """PYP with invalid d/alpha raises ParameterError."""
        with pytest.raises(ParameterError):
            PYP(d=d, alpha=alpha)

    @pytest.mark.parametrize(
        "h,gamma",
        [
            (-1.0, 0.5),  # h <= 0
            (0.0, 0.5),  # h == 0
            (2.0, 1.5),  # gamma > 1
            (2.0, -0.1),  # gamma < 0
        ],
    )
    def test_init_invalid_h_gamma(self, h, gamma):
        """PYP with invalid h/gamma raises ParameterError."""
        with pytest.raises(ParameterError):
            PYP(h=h, gamma=gamma)

    def test_properties(self):
        """PYP property calculations."""
        pyp = PYP(d=0.5, alpha=1.0)
        assert np.isfinite(pyp.h) and 0 <= pyp.gamma <= 1

    def test_methods(self):
        """PYP method calculations."""
        pyp = PYP(d=0.5, alpha=1.0)
        n = np.array([1, 10, 100])
        assert_bounded(pyp.uniqueness(n))
        assert_bounded(pyp.correctness(n))
        assert 0 <= pyp.kanon_violations(n[0], k=2) <= 1

    @pytest.mark.parametrize("n,k", [(10, 2), (100, 5), (1000, 10)])
    def test_kanon_violations_ranges(self, n, k):
        """k-anonymity violations for different n,k combinations."""
        assert 0 <= PYP(d=0.5, alpha=1.0).kanon_violations(n, k) <= 1


# =============================================================================
# Unit Tests: FLModel Class
# =============================================================================


class TestFLModel:
    """Tests for FLModel class."""

    def test_init(self):
        """FLModel initialization."""
        assert FLModel(2.0).h == 2.0

    @pytest.mark.parametrize("h,expected", [(-1.0, 0.0), (150.0, 100.0)])
    def test_init_clipping(self, h, expected):
        """Entropy parameter clipping."""
        assert FLModel(h).h == expected

    def test_methods(self):
        """FLModel method calculations."""
        model = FLModel(2.0)
        n = np.array([1, 10, 100])
        assert_bounded(model.uniqueness(n))
        assert_bounded(model.correctness(n))
        assert np.all(np.diff(model.uniqueness(n)) <= 0)
        assert np.all(np.diff(model.correctness(n)) <= 0)

    def test_kanon_violations_not_implemented(self):
        """k-anonymity violations raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            FLModel(2.0).kanon_violations(10, 2)
