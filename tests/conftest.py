"""
Shared test configuration, fixtures, and Hypothesis strategies.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import strategies as st


# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def pyp_params(draw):
    """Generate valid PYP (d, α) parameter pairs satisfying α > -d."""
    d = draw(st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False))
    α = draw(st.floats(min_value=max(0.01, -d + 0.01), max_value=100.0, allow_nan=False, allow_infinity=False))
    return {"d": d, "α": α}


frequency_arrays = st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=100).map(
    lambda x: np.array(sorted(x))
)

sample_arrays = st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=500).map(np.array)

sample_size_arrays = st.lists(st.integers(min_value=1, max_value=10**10), min_size=1, max_size=20).map(
    lambda x: np.array(sorted(x))
)

entropy_values = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)


@st.composite
def training_data_strategy(draw):
    """Generate valid training data for extrapolation models."""
    n_points = draw(st.integers(min_value=3, max_value=10))
    base_sizes = sorted(
        draw(
            st.lists(
                st.integers(min_value=10, max_value=10000),
                min_size=n_points,
                max_size=n_points,
                unique=True,
            )
        )
    )
    kappas = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        reverse=True,
    )
    return pd.DataFrame({"n": base_sizes, "κ": kappas})


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_bounded(values, low=0.0, high=1.0, atol=1e-10):
    """Assert all values are within [low, high] bounds."""
    assert np.all(values >= low - atol), f"Values below {low}: {values[values < low - atol]}"
    assert np.all(values <= high + atol), f"Values above {high}: {values[values > high + atol]}"


def assert_monotonic_decreasing(values, atol=1e-10):
    """Assert values are monotonically decreasing."""
    assert np.all(np.diff(values) <= atol), f"Not monotonically decreasing: {values}"


def assert_valid_predictions(predictions, monotonic=True):
    """Assert predictions are bounded in [0,1] and optionally monotonically decreasing."""
    assert_bounded(predictions)
    if monotonic:
        assert_monotonic_decreasing(predictions)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_sizes():
    """Standard sample sizes for testing."""
    return np.array([1, 10, 100, 1000])


@pytest.fixture(scope="session")
def training_data():
    """Standard training data for models."""
    return pd.DataFrame({"n": [10, 100, 1000], "κ": [0.8, 0.5, 0.3]})


@pytest.fixture(scope="session")
def sample_frequencies():
    """Sample frequency data."""
    return np.array([1, 1, 2, 3, 4])


@pytest.fixture(scope="session")
def large_array():
    """Large array for performance testing."""
    return np.random.default_rng(42).integers(0, 1000, size=10000)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
