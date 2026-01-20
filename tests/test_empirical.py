import numpy as np
import pandas as pd
import pytest
from dataless.empirical import correctness, counts_from_dataframe, empirical_entropy, frequencies, uniqueness
from numpy.testing import assert_allclose, assert_array_equal


class TestFrequencies:
    """Tests for the frequencies function."""

    def test_basic_array(self):
        """Test frequencies with basic input array."""
        arr = np.array([1, 2, 2, 3, 3, 3])
        expected = np.array([1, 2, 3])
        assert_array_equal(frequencies(arr), expected)

    def test_single_element(self):
        """Test frequencies with array containing single unique element."""
        arr = np.array([1, 1, 1])
        expected = np.array([3])
        assert_array_equal(frequencies(arr), expected)

    def test_all_unique(self):
        """Test frequencies with array containing all unique elements."""
        arr = np.array([1, 2, 3])
        expected = np.array([1, 1, 1])
        assert_array_equal(frequencies(arr), expected)

    def test_empty_array(self):
        """Test frequencies with empty array."""
        arr = np.array([])
        expected = np.array([])
        assert_array_equal(frequencies(arr), expected)

    def test_non_sequential(self):
        """Test frequencies with non-sequential values."""
        arr = np.array([10, 20, 10, 30, 20, 10])
        expected = np.array([1, 2, 3])
        assert_array_equal(frequencies(arr), expected)


class TestEmpiricalEntropy:
    """Tests for the empirical_entropy function."""

    def test_uniform_distribution(self):
        """Test entropy of uniform distribution."""
        arr = np.array([1, 2, 3, 4])
        expected = np.log(4)  # ln(4) for base e
        assert_allclose(empirical_entropy(arr), expected)

    def test_single_value(self):
        """Test entropy of single value (should be 0)."""
        arr = np.array([1, 1, 1])
        assert_allclose(empirical_entropy(arr), 0.0)

    def test_binary_distribution(self):
        """Test entropy of binary distribution."""
        arr = np.array([0, 0, 0, 1])  # 75% zeros, 25% ones
        expected = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
        assert_allclose(empirical_entropy(arr), expected)

    def test_different_base(self):
        """Test entropy with different logarithm base."""
        arr = np.array([1, 2, 3, 4])
        expected = np.log2(4)  # log2(4) for base 2
        assert_allclose(empirical_entropy(arr, base=2), expected)

    def test_empty_array(self):
        """Test entropy of empty array raises error."""
        with pytest.raises(ValueError):
            empirical_entropy(np.array([]))


class TestCountsFromDataframe:
    """Tests for the counts_from_dataframe function."""

    def test_simple_dataframe(self):
        """Test counts from simple DataFrame."""
        df = pd.DataFrame(
            {
                "A": [1, 1, 2],
                "B": [1, 1, 2],
            },
        )
        expected = np.array([1, 2])
        assert_array_equal(counts_from_dataframe(df), expected)

    def test_all_unique_rows(self):
        """Test counts when all rows are unique."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
        )
        expected = np.array([1, 1, 1])
        assert_array_equal(counts_from_dataframe(df), expected)

    def test_all_identical_rows(self):
        """Test counts when all rows are identical."""
        df = pd.DataFrame(
            {
                "A": [1, 1, 1],
                "B": [2, 2, 2],
            },
        )
        expected = np.array([3])
        assert_array_equal(counts_from_dataframe(df), expected)

    def test_empty_dataframe(self):
        """Test counts from empty DataFrame."""
        df = pd.DataFrame({"A": [], "B": []})
        expected = np.array([])
        assert_array_equal(counts_from_dataframe(df), expected)


class TestUniqueness:
    """Tests for the uniqueness function."""

    def test_basic_frequencies(self):
        """Test uniqueness with basic frequency distribution."""
        freqs = np.array([1, 1, 2, 3])  # 2 unique elements out of 7 total
        expected = 2 / 7
        assert_allclose(uniqueness(freqs), expected)

    def test_all_unique(self):
        """Test uniqueness when all elements appear once."""
        freqs = np.array([1, 1, 1])
        assert_allclose(uniqueness(freqs), 1.0)

    def test_no_unique(self):
        """Test uniqueness when no elements appear once."""
        freqs = np.array([2, 3, 4])
        assert_allclose(uniqueness(freqs), 0.0)

    def test_empty_array(self):
        """Test uniqueness of empty array."""
        with pytest.raises(ValueError):
            uniqueness(np.array([]))

    def test_null_array(self):
        """Test uniqueness of array with zero total frequency."""
        with pytest.raises(ValueError):
            uniqueness(np.array([0, 0, 0]))


class TestCorrectness:
    """Tests for the correctness function."""

    def test_basic_frequencies(self):
        """Test correctness with basic frequency distribution."""
        freqs = np.array([1, 2, 3])  # 3 unique elements out of 6 total
        expected = 3 / 6
        assert_allclose(correctness(freqs), expected)

    def test_all_unique(self):
        """Test correctness when all elements appear once."""
        freqs = np.array([1, 1, 1])
        assert_allclose(correctness(freqs), 1.0)

    def test_single_element(self):
        """Test correctness with single frequency."""
        freqs = np.array([5])  # 1 unique element out of 5 total
        expected = 1 / 5
        assert_allclose(correctness(freqs), expected)

    def test_empty_array(self):
        """Test correctness of empty array."""
        with pytest.raises(ValueError):
            correctness(np.array([]))

    def test_null_array(self):
        """Test correctness of array with zero total frequency."""
        with pytest.raises(ValueError):
            correctness(np.array([0, 0, 0]))
