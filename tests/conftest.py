"""
Shared test configuration and fixtures.

This module provides pytest configuration and shared fixtures
for all test modules in the project.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_sizes():
    """Fixture providing standard sample sizes for testing."""
    return np.array([1, 10, 100, 1000])


@pytest.fixture(scope="session")
def training_data():
    """Fixture providing standard training data for models."""
    return pd.DataFrame({
        'n': [10, 100, 1000],
        'κ': [0.8, 0.5, 0.3]
    })


@pytest.fixture(scope="session")
def sample_frequencies():
    """Fixture providing sample frequency data."""
    return np.array([1, 1, 2, 3, 4])


@pytest.fixture(scope="session")
def sample_multiplicities():
    """Fixture providing sample multiplicity data."""
    mm = np.array([2, 1, 1, 1])  # multiplicities
    icts = np.array([1, 2, 3, 4])  # counts
    return mm, icts


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
