import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

from .exceptions import OptimizationWarning, TrainingDataError
from .model import PYP, FLModel

MetricType = Literal["correctness", "uniqueness"]


class PYPParams(NamedTuple):
    h: float
    gamma: float


class FLParams(NamedTuple):
    h: float


class ExpDecayParams(NamedTuple):
    a: float
    b: float


class PolynomialParams(NamedTuple):
    a: float
    b: float
    c: float


class AbstractExtrapolation(ABC):
    """Abstract base class for extrapolation models.

    Defines the interface for models that can be trained on empirical data
    and used to predict scaling behavior at new sample sizes.
    """

    def __init__(
        self,
        n: ndarray,
        *,
        correctness: ndarray | None = None,
        uniqueness: ndarray | None = None,
    ) -> None:
        """Initialize the model with training data.

        Args:
            n: Array of sample sizes (or DataFrame with 'n' and 'κ' columns, deprecated)
            correctness: Array of correctness values (mutually exclusive with uniqueness)
            uniqueness: Array of uniqueness values (mutually exclusive with correctness)

        """
        # Backward compatibility: accept DataFrame with 'n' and 'κ' columns
        if hasattr(n, "columns") and "n" in n.columns and "κ" in n.columns:
            warnings.warn(
                "Passing a DataFrame is deprecated. Use Model(n, correctness=values) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            correctness = n["κ"].values
            n = n["n"].values

        self._init_training_data(n, correctness, uniqueness)
        self.train()

    def _init_training_data(self, n: ndarray, correctness: ndarray | None, uniqueness: ndarray | None) -> None:
        if correctness is not None and uniqueness is not None:
            raise TrainingDataError("Provide either correctness or uniqueness, not both")
        if correctness is None and uniqueness is None:
            raise TrainingDataError("Must provide either correctness or uniqueness")

        self.n_training = np.asarray(n)
        if correctness is not None:
            self.metric: MetricType = "correctness"
            self.values_training = np.asarray(correctness)
        else:
            self.metric = "uniqueness"
            self.values_training = np.asarray(uniqueness)
        self.validate_training_data(self.n_training, self.values_training, self.metric)

    def validate_training_data(self, n: ndarray, values: ndarray, metric: MetricType) -> None:
        """Validate training data arrays."""
        if len(n) != len(values):
            raise TrainingDataError("n and values must have same length")
        if len(n) < 3:
            raise TrainingDataError("Training data must have at least 3 samples")
        if not np.all((values >= 0) & (values <= 1)):
            raise TrainingDataError(f"All {metric} values must be between 0 and 1")
        if not np.all(n > 0):
            raise TrainingDataError("All n values must be positive")

    # Subclasses must define INIT_STATE and PARAMS_CLASS
    INIT_STATE: tuple
    PARAMS_CLASS: type[NamedTuple]

    def __getattr__(self, name: str):
        """Forward parameter access to self.params for backward compatibility."""
        if name != "params" and hasattr(self, "params") and hasattr(self.params, name):
            return getattr(self.params, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @abstractmethod
    def _make_loss_fn(self) -> Callable:
        """Create the loss function for optimization."""

    def train(self) -> None:
        """Train the model on the provided training data."""
        loss_fn = self._make_loss_fn()
        self.params = self.PARAMS_CLASS(*self._run_optimization(loss_fn, self.INIT_STATE))

    @abstractmethod
    def predict_correctness(self, n: int | ndarray) -> ndarray:
        """Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values for the given sample sizes

        """

    @abstractmethod
    def predict_uniqueness(self, n: int | ndarray) -> ndarray:
        """Predict uniqueness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted uniqueness values for the given sample sizes

        """

    def predict(self, n: int | ndarray) -> ndarray:
        """Predict values for new sample sizes based on trained metric.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted uniqueness/correctness values for the given sample sizes
        """

        if self.metric == "correctness":
            return self.predict_correctness(n)
        elif self.metric == "uniqueness":
            return self.predict_uniqueness(n)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def test(self, n: int | ndarray) -> ndarray:
        """Deprecated: Use predict() instead."""
        warnings.warn(
            "test() is deprecated and will be removed in a future version. Use predict() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.predict(n)

    def _run_optimization(self, loss_fn: Callable, init_state: tuple | list) -> ndarray:
        res = minimize(loss_fn, init_state, method="Nelder-Mead")
        if not res.success:
            warnings.warn(f"Optimization did not converge: {res.message}", OptimizationWarning, stacklevel=3)
        self._optimization_result = res
        return res.x

    # Class attributes for summary() - override in subclasses
    MODEL_NAME: str = "Extrapolation Model"

    def _get_param_str(self) -> str:
        """Return formatted parameter string for summary."""
        return ", ".join(f"{name}={getattr(self.params, name):.4f}" for name in self.params._fields)

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            str: Model summary including parameters and training info

        """
        header = self.MODEL_NAME
        n_min, n_max = self.n_training.min(), self.n_training.max()
        return (
            f"{header}\n{'=' * len(header)}\n"
            f"Trained on: {self.metric}\n"
            f"Parameters: {self._get_param_str()}\n"
            f"Training points: {len(self.n_training)}\n"
            f"Training range: n in [{n_min:,}, {n_max:,}]"
        )


class StatisticalModelMixin:
    """Mixin for statistical models that can predict both metrics from fitted parameters."""

    @abstractmethod
    def _get_model(self):
        """Return the underlying statistical model instance."""

    def _predict_with_model(self, method: str, n):
        n_array = np.atleast_1d(n)
        result = getattr(self._get_model(), method)(n_array)
        return result[0] if np.isscalar(n) else result

    def predict_correctness(self, n: int | ndarray) -> ndarray:
        """Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values

        """
        return self._predict_with_model("correctness", n)

    def predict_uniqueness(self, n: int | ndarray) -> ndarray:
        """Predict uniqueness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted uniqueness values

        """
        return self._predict_with_model("uniqueness", n)


class PYPExtrapolation(StatisticalModelMixin, AbstractExtrapolation):
    """Extrapolation model based on Pitman-Yor Process.

    This model fits a Pitman-Yor Process to training data and uses it
    to predict scaling behavior at new sample sizes.
    """

    INIT_STATE = (12, 0.26)
    PARAMS_CLASS = PYPParams
    MODEL_NAME = "PYP Extrapolation Model"

    def _make_loss_fn(self) -> Callable:
        n_range, emp_values, metric = self.n_training, self.values_training, self.metric

        def loss_fn(x):
            h, gamma = x[0], x[1]
            if h <= 0 or gamma < 0 or gamma > 1:
                return 1e10
            try:
                expected = getattr(PYP(h=h, gamma=gamma), metric)(n_range)
                return (np.log(n_range) * (expected - emp_values) ** 2).mean()
            except Exception:
                return 1e10

        return loss_fn

    def _get_model(self):
        return PYP(h=self.h, gamma=self.gamma)


class FLExtrapolation(StatisticalModelMixin, AbstractExtrapolation):
    """Extrapolation model based on baseline entropy model.

    This model provides a simpler alternative to PYP extrapolation,
    using only entropy-based calculations.
    """

    INIT_STATE = (12.0,)
    PARAMS_CLASS = FLParams
    MODEL_NAME = "FL Extrapolation Model"

    def _make_loss_fn(self) -> Callable:
        n_range, emp_values, metric = self.n_training, self.values_training, self.metric

        def loss_fn(x):
            expected = getattr(FLModel(h=float(x[0])), metric)(n_range)
            return (np.log(n_range) * (expected - emp_values) ** 2).mean()

        return loss_fn

    def _get_model(self):
        return FLModel(h=self.h)


class CurveFitExtrapolationMixin:
    """Mixin for curve-fitting models that can only predict their trained metric."""

    metric: MetricType

    @staticmethod
    @abstractmethod
    def _compute_static(params: tuple, n: ndarray) -> ndarray:
        """Compute raw prediction from parameters (subclass implements)."""

    def _compute(self, n: ndarray) -> ndarray:
        """Compute raw prediction using fitted parameters."""
        result = self._compute_static(tuple(self.params), n)
        return np.clip(result, 0, 1)

    def _make_loss_fn(self) -> Callable:
        """Create loss function for curve-fitting optimization."""

        def loss_fn(x):
            est_values = self._compute_static(tuple(x), self.n_training)
            return (np.log(self.n_training) * (est_values - self.values_training) ** 2).mean()

        return loss_fn

    def predict_correctness(self, n):
        if self.metric != "correctness":
            raise NotImplementedError(
                f"This model was trained on {self.metric}. "
                f"Use predict_{self.metric}() or train a new model on correctness data."
            )
        return self._compute(n)

    def predict_uniqueness(self, n):
        if self.metric != "uniqueness":
            raise NotImplementedError(
                f"This model was trained on {self.metric}. "
                f"Use predict_{self.metric}() or train a new model on uniqueness data."
            )
        return self._compute(n)


class ExpDecayExtrapolation(CurveFitExtrapolationMixin, AbstractExtrapolation):
    """Exponential decay-based extrapolation model.

    This model fits an exponential decay curve to the training data
    for predicting scaling behavior. Can be trained on either correctness
    or uniqueness, but can only predict the metric it was trained on.
    """

    INIT_STATE = (1, 1)
    PARAMS_CLASS = ExpDecayParams
    MODEL_NAME = "ExpDecay Extrapolation Model"

    @staticmethod
    def _compute_static(params: tuple, n: ndarray) -> ndarray:
        a, b = params
        return a * np.exp(-b * np.sqrt(n)) + (1 - a * np.exp(-b))


class PolynomialExtrapolation(CurveFitExtrapolationMixin, AbstractExtrapolation):
    """Polynomial fit-based extrapolation model.

    This model fits a third-degree polynomial to the log-transformed data
    for predicting scaling behavior. Can be trained on either correctness
    or uniqueness, but can only predict the metric it was trained on.
    """

    INIT_STATE = (0, -1, -1)
    PARAMS_CLASS = PolynomialParams
    MODEL_NAME = "Polynomial Extrapolation Model"

    @staticmethod
    def _compute_static(params: tuple, n: ndarray) -> ndarray:
        a, b, c = params
        log_n = np.log10(n)
        return a * log_n**3 + b * log_n**2 + c * log_n + 1
