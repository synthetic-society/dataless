"""Custom exceptions for the dataless package."""


class DatalessError(Exception):
    """Base exception for all dataless errors."""


class TrainingDataError(DatalessError):
    """Raised when training data is invalid or insufficient."""


class OptimizationWarning(UserWarning):
    """Warning issued when model optimization does not fully converge."""


class OptimizationError(DatalessError):
    """Raised when model optimization fails to converge."""


class ParameterError(DatalessError):
    """Raised when model parameters are invalid."""
