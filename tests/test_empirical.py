"""Unit tests for empirical statistics functions."""

import numpy as np
import pandas as pd
import pytest
from dataless.empirical import correctness, counts_from_dataframe, empirical_entropy, frequencies, uniqueness
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from conftest import assert_bounded, frequency_arrays, sample_arrays

# =============================================================================
# Property-Based Tests
# =============================================================================


class TestFrequencies:
    """Tests for the frequencies function."""

    @given(sample_arrays)
    def test_sum_equals_input_length(self, arr):
        """Sum of frequencies equals input length."""
        assert frequencies(arr).sum() == len(arr)

    @given(sample_arrays)
    def test_always_sorted_ascending(self, arr):
        """Frequencies are always sorted in ascending order."""
        freqs = frequencies(arr)
        assert np.all(np.diff(freqs) >= 0)

    @given(sample_arrays)
    def test_all_positive(self, arr):
        """All frequencies are positive integers."""
        assert np.all(frequencies(arr) >= 1)

    @given(st.lists(st.integers(0, 100), min_size=1, max_size=100, unique=True).map(np.array))
    def test_all_unique_elements_give_ones(self, arr):
        """Array with all unique elements has all frequencies = 1."""
        freqs = frequencies(arr)
        assert np.all(freqs == 1) and len(freqs) == len(arr)

    @pytest.mark.parametrize(
        "arr,expected",
        [
            (np.array([1, 2, 2, 3, 3, 3]), np.array([1, 2, 3])),
            (np.array([1, 1, 1]), np.array([3])),
            (np.array([1, 2, 3]), np.array([1, 1, 1])),
            (np.array([]), np.array([])),
            (np.array([10, 20, 10, 30, 20, 10]), np.array([1, 2, 3])),
        ],
    )
    def test_expected_values(self, arr, expected):
        """Test specific input/output pairs."""
        assert_array_equal(frequencies(arr), expected)


class TestEmpiricalEntropy:
    """Tests for the empirical_entropy function."""

    @given(sample_arrays)
    def test_non_negative(self, arr):
        """Entropy is always non-negative."""
        assert empirical_entropy(arr) >= 0

    @given(sample_arrays)
    def test_bounded_by_log_n(self, arr):
        """Entropy ≤ log(number of unique elements)."""
        h = empirical_entropy(arr)
        assert h <= np.log(len(np.unique(arr))) + 1e-10

    @given(st.integers(min_value=1, max_value=1000))
    def test_single_value_has_zero_entropy(self, value):
        """Array with single unique value has zero entropy."""
        assert_allclose(empirical_entropy(np.array([value] * 10)), 0.0)

    @given(st.lists(st.integers(0, 100), min_size=2, max_size=100, unique=True).map(np.array))
    def test_uniform_distribution_has_max_entropy(self, arr):
        """Uniform distribution achieves maximum entropy."""
        assert_allclose(empirical_entropy(arr), np.log(len(arr)))

    def test_empty_array_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError):
            empirical_entropy(np.array([]))

    def test_different_base(self):
        """Entropy with different logarithm base."""
        arr = np.array([1, 2, 3, 4])
        assert_allclose(empirical_entropy(arr, base=2), np.log2(4))


class TestUniqueness:
    """Tests for the uniqueness function."""

    @given(frequency_arrays)
    def test_bounded_zero_one(self, freqs):
        """Uniqueness is always in [0, 1]."""
        assert_bounded(uniqueness(freqs))

    @given(st.integers(min_value=1, max_value=100))
    def test_all_singletons_equals_one(self, n):
        """Array of all 1s has uniqueness = 1."""
        assert_allclose(uniqueness(np.ones(n, dtype=int)), 1.0)

    @given(st.lists(st.integers(min_value=2, max_value=100), min_size=1, max_size=50).map(np.array))
    def test_no_singletons_equals_zero(self, freqs):
        """Array with no 1s has uniqueness = 0."""
        assert_allclose(uniqueness(freqs), 0.0)

    @pytest.mark.parametrize(
        "freqs,expected",
        [
            (np.array([1, 1, 2, 3]), 2 / 7),
            (np.array([1, 1, 1]), 1.0),
            (np.array([2, 3, 4]), 0.0),
        ],
    )
    def test_expected_values(self, freqs, expected):
        """Test specific input/output pairs."""
        assert_allclose(uniqueness(freqs), expected)

    def test_empty_array_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError):
            uniqueness(np.array([]))

    def test_zero_total_raises(self):
        """Array with zero total frequency raises ValueError."""
        with pytest.raises(ValueError):
            uniqueness(np.array([0, 0, 0]))


class TestCorrectness:
    """Tests for the correctness function."""

    @given(frequency_arrays)
    def test_bounded_zero_one(self, freqs):
        """Correctness is always in [0, 1]."""
        assert_bounded(correctness(freqs))

    @given(st.integers(min_value=1, max_value=100))
    def test_all_singletons_equals_one(self, n):
        """Array of all 1s has correctness = 1."""
        assert_allclose(correctness(np.ones(n, dtype=int)), 1.0)

    @given(frequency_arrays)
    def test_equals_num_unique_over_total(self, freqs):
        """Correctness = len(freqs) / sum(freqs)."""
        assert_allclose(correctness(freqs), len(freqs) / freqs.sum())

    @pytest.mark.parametrize(
        "freqs,expected",
        [
            (np.array([1, 2, 3]), 3 / 6),
            (np.array([1, 1, 1]), 1.0),
            (np.array([5]), 1 / 5),
        ],
    )
    def test_expected_values(self, freqs, expected):
        """Test specific input/output pairs."""
        assert_allclose(correctness(freqs), expected)

    def test_empty_array_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError):
            correctness(np.array([]))

    def test_zero_total_raises(self):
        """Array with zero total frequency raises ValueError."""
        with pytest.raises(ValueError):
            correctness(np.array([0, 0, 0]))


class TestCountsFromDataframe:
    """Tests for the counts_from_dataframe function."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            ({"A": [1, 1, 2], "B": [1, 1, 2]}, np.array([1, 2])),
            ({"A": [1, 2, 3], "B": [4, 5, 6]}, np.array([1, 1, 1])),
            ({"A": [1, 1, 1], "B": [2, 2, 2]}, np.array([3])),
            ({"A": [], "B": []}, np.array([])),
        ],
    )
    def test_expected_values(self, data, expected):
        """Test specific input/output pairs."""
        assert_array_equal(counts_from_dataframe(pd.DataFrame(data)), expected)
