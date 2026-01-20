from abc import ABC, abstractmethod
from collections.abc import Callable
import warnings

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.optimize import minimize

from .exceptions import OptimizationWarning, TrainingDataError
from .model import PYP, FLModel


class AbstractExtrapolation(ABC):
    """Abstract base class for extrapolation models.

    Defines the interface for models that can be trained on empirical data
    and used to predict scaling behavior at new sample sizes.
    """

    @abstractmethod
    def __init__(self, training_data: pd.DataFrame) -> None:
        """Initialize the extrapolation model.

        Args:
            training_data: DataFrame containing training samples

        """

    def validate_training_data(self, dd: pd.DataFrame) -> None:
        """Validate the training data.

        Args:
            dd: DataFrame containing training data

        Raises:
            TrainingDataError: If training data is invalid

        """
        if not isinstance(dd, pd.DataFrame):
            raise TrainingDataError("Training data must be a DataFrame")
        if not all(col in dd.columns for col in ["n", "κ"]):
            raise TrainingDataError("Training data must have columns 'n' and 'κ'")

        if len(dd) < 3:
            raise TrainingDataError("Training data must have at least 3 samples")

        if not all(dd["κ"].between(0, 1)):
            raise TrainingDataError("All κ values must be between 0 and 1")

        if not all(dd["n"] > 0):
            raise TrainingDataError("All n values must be positive")

    @classmethod
    @abstractmethod
    def make_loss_fun(cls, dd: pd.DataFrame) -> Callable:
        """Loss function for model optimization.

        Args:
            dd: DataFrame containing training data

        Returns:
            Callable: Loss function for optimization

        """

    @abstractmethod
    def train(self) -> None:
        """Train the model on the provided training data."""

    @abstractmethod
    def predict(self, n: int | ndarray) -> ndarray:
        """Predict values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted values for the given sample sizes

        """

    def test(self, n: int | ndarray) -> ndarray:
        """Deprecated: Use predict() instead.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted values for the given sample sizes

        """
        warnings.warn(
            "test() is deprecated and will be removed in a future version. Use predict() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.predict(n)

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            str: Model summary including parameters and training info

        """


class PYPExtrapolation(AbstractExtrapolation):
    """Extrapolation model based on Pitman-Yor Process.

    This model fits a Pitman-Yor Process to training data and uses it
    to predict scaling behavior at new sample sizes.
    """

    INIT_STATE = (12, 0.26)  # Initial state for optimization (h, γ)

    def __init__(self, training_data: pd.DataFrame) -> None:
        """Initialize PYP extrapolation model.

        Args:
            training_data: DataFrame with columns 'n' (sample sizes) and 'κ' (observations)

        """
        self.training_data = training_data
        self.validate_training_data(training_data)

        self.train()

    @staticmethod
    def make_loss_fun(dd: pd.DataFrame) -> Callable:
        """Create loss function for PYP parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization

        """
        n_range = dd.n.to_numpy()
        emp_κ = dd["κ"].values

        def loss_fun(x):
            # Return large value for invalid parameter combinations during optimization
            h, γ = x[0], x[1]
            if h <= 0 or γ < 0 or γ > 1:
                return 1e10
            try:
                expected_κ = PYP(h=h, γ=γ).correctness(n_range)
                return (np.log(n_range) * (expected_κ - emp_κ) ** 2).mean()
            except Exception:
                return 1e10

        return loss_fun

    def train(self) -> None:
        """Train the model by optimizing PYP parameters.

        Uses Nelder-Mead optimization to find optimal h and γ values.

        Warns:
            OptimizationWarning: If optimization does not converge (as warning)

        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, type(self).INIT_STATE, method="Nelder-Mead")
        if not res.success:
            warnings.warn(
                f"Optimization did not converge: {res.message}",
                OptimizationWarning,
                stacklevel=2,
            )
        self.h, self.γ = res.x
        self._optimization_result = res

    def predict(self, n: int | ndarray) -> ndarray:
        """Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values

        """
        n_array = np.atleast_1d(n)
        result = PYP(h=self.h, γ=self.γ).correctness(n_array)
        return result[0] if np.isscalar(n) else result

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            str: Model summary including parameters and training info

        """
        n_min = self.training_data["n"].min()
        n_max = self.training_data["n"].max()
        return (
            f"PYP Extrapolation Model\n"
            f"=======================\n"
            f"Parameters: h={self.h:.4f}, γ={self.γ:.4f}\n"
            f"Training points: {len(self.training_data)}\n"
            f"Training range: n ∈ [{n_min:,}, {n_max:,}]"
        )


class FLExtrapolation(AbstractExtrapolation):
    """Extrapolation model based on baseline entropy model.

    This model provides a simpler alternative to PYP extrapolation,
    using only entropy-based calculations.
    """

    INIT_STATE = [12.0]  # Initial entropy value for optimization

    def __init__(self, training_data):
        """Initialize FL extrapolation model.

        Args:
            training_data: DataFrame with columns 'n' and 'κ'

        """
        self.training_data = training_data
        self.validate_training_data(training_data)

        self.train()

    @classmethod
    def make_loss_fun(cls, dd):
        """Loss function for entropy parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization

        """
        n_range = dd.n.to_numpy()
        emp_κ = dd["κ"].to_numpy()

        def loss_fun(x):
            expected_κ = FLModel(h=float(x[0])).correctness(n_range)
            return (np.log(n_range) * (expected_κ - emp_κ) ** 2).mean()

        return loss_fun

    def train(self):
        """Train the model by optimizing the entropy parameter.

        Uses Nelder-Mead optimization to find optimal entropy value.
        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, self.INIT_STATE, method="Nelder-Mead")
        if not res.success:
            warnings.warn(
                f"Optimization did not converge: {res.message}",
                OptimizationWarning,
                stacklevel=2,
            )
        self.h = float(res.x[0])
        self._optimization_result = res

    def predict(self, n):
        """Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values

        """
        n_array = np.atleast_1d(n)
        result = FLModel(h=self.h).correctness(n_array)
        return result[0] if np.isscalar(n) else result

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            str: Model summary including parameters and training info

        """
        n_min = self.training_data["n"].min()
        n_max = self.training_data["n"].max()
        return (
            f"FL Extrapolation Model\n"
            f"======================\n"
            f"Parameters: h={self.h:.4f}\n"
            f"Training points: {len(self.training_data)}\n"
            f"Training range: n ∈ [{n_min:,}, {n_max:,}]"
        )


class ExpDecayExtrapolation(AbstractExtrapolation):
    """Exponential decay-based extrapolation model.

    This model fits an exponential decay curve to the training data
    for predicting scaling behavior.
    """

    def __init__(self, training_data):
        """Initialize exponential decay model.

        Args:
            training_data: DataFrame with columns 'n' and 'κ'

        """
        self.training_data = training_data
        self.validate_training_data(training_data)

        self.train()

    @staticmethod
    def correctness(a, b, n):
        """Compute correctness using exponential decay formula.

        Args:
            a: Amplitude parameter
            b: Decay rate parameter
            n: Sample size(s)

        Returns:
            ndarray: Correctness values

        """
        return a * np.exp(-b * np.sqrt(n)) + (1 - a * np.exp(-b))

    @classmethod
    def make_loss_fun(cls, dd):
        """Loss function for exponential decay parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization

        """
        n_range = dd.n.to_numpy()
        emp_κ = dd["κ"].to_numpy()

        def loss_fun(x):
            est_κ = cls.correctness(x[0], x[1], n_range)
            return (np.log(n_range) * (est_κ - emp_κ) ** 2).mean()

        return loss_fun

    def train(self):
        """Train the model by optimizing exponential decay parameters.

        Uses Nelder-Mead optimization to find optimal a and b values.
        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, (1, 1), method="Nelder-Mead")
        if not res.success:
            warnings.warn(
                f"Optimization did not converge: {res.message}",
                OptimizationWarning,
                stacklevel=2,
            )
        self.a, self.b = res.x
        self._optimization_result = res

    def predict(self, n):
        """Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values, clipped to [0,1]

        """
        n_array = np.atleast_1d(n)
        result = np.clip(type(self).correctness(self.a, self.b, n_array), 0, 1)
        return result[0] if np.isscalar(n) else result

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            str: Model summary including parameters and training info

        """
        n_min = self.training_data["n"].min()
        n_max = self.training_data["n"].max()
        return (
            f"Exponential Decay Extrapolation Model\n"
            f"=====================================\n"
            f"Parameters: a={self.a:.4f}, b={self.b:.4f}\n"
            f"Training points: {len(self.training_data)}\n"
            f"Training range: n ∈ [{n_min:,}, {n_max:,}]"
        )


class PolynomialExtrapolation(AbstractExtrapolation):
    """Polynomial fit-based extrapolation model.

    This model fits a third-degree polynomial to the log-transformed data
    for predicting scaling behavior.
    """

    def __init__(self, training_data):
        """Initialize polynomial extrapolation model.

        Args:
            training_data: DataFrame with columns 'n' and 'κ'

        """
        self.training_data = training_data
        self.validate_training_data(training_data)

        self.train()

    @staticmethod
    def correctness(a, b, c, n):
        """Compute correctness using polynomial formula.

        Args:
            a: Cubic coefficient
            b: Quadratic coefficient
            c: Linear coefficient
            n: Sample size(s)

        Returns:
            ndarray: Correctness values

        """
        return a * np.log10(n) ** 3 + b * np.log10(n) ** 2 + c * np.log10(n) + 1

    @classmethod
    def make_loss_fun(cls, dd):
        """Loss function for polynomial parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization

        """
        n_range = dd.n.to_numpy()
        emp_κ = dd["κ"].to_numpy()

        def loss_fun(x):
            est_κ = cls.correctness(x[0], x[1], x[2], n_range)
            return (np.log(n_range) * (est_κ - emp_κ) ** 2).mean()

        return loss_fun

    def train(self):
        """Train the model by optimizing polynomial coefficients.

        Uses Nelder-Mead optimization to find optimal polynomial coefficients.
        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, (0, -1, -1), method="Nelder-Mead")
        if not res.success:
            warnings.warn(
                f"Optimization did not converge: {res.message}",
                OptimizationWarning,
                stacklevel=2,
            )
        self.a, self.b, self.c = res.x
        self._optimization_result = res

    def predict(self, n):
        """Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values, clipped to [0,1]

        """
        n_array = np.atleast_1d(n)
        result = np.clip(type(self).correctness(self.a, self.b, self.c, n_array), 0, 1)
        return result[0] if np.isscalar(n) else result

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            str: Model summary including parameters and training info

        """
        n_min = self.training_data["n"].min()
        n_max = self.training_data["n"].max()
        return (
            f"Polynomial Extrapolation Model\n"
            f"==============================\n"
            f"Parameters: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}\n"
            f"Training points: {len(self.training_data)}\n"
            f"Training range: n ∈ [{n_min:,}, {n_max:,}]"
        )
