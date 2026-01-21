"""Dataless: Modeling and forecasting identification techniques at scale."""

from dataless.empirical import (
    correctness,
    counts_from_dataframe,
    empirical_entropy,
    frequencies,
    uniqueness,
)
from dataless.exceptions import (
    DatalessError,
    OptimizationError,
    OptimizationWarning,
    ParameterError,
    TrainingDataError,
)
from dataless.extrapolate import (
    ExpDecayExtrapolation,
    FLExtrapolation,
    PolynomialExtrapolation,
    PYPExtrapolation,
)
from dataless.model import PYP, FLModel

__version__ = "0.1.0"

__all__ = [
    # Extrapolation models
    "PYPExtrapolation",
    "FLExtrapolation",
    "ExpDecayExtrapolation",
    "PolynomialExtrapolation",
    # Statistical models
    "PYP",
    "FLModel",
    # Empirical functions
    "frequencies",
    "correctness",
    "uniqueness",
    "empirical_entropy",
    "counts_from_dataframe",
    # Exceptions and Warnings
    "DatalessError",
    "TrainingDataError",
    "OptimizationError",
    "OptimizationWarning",
    "ParameterError",
]
