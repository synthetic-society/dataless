from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.optimize import minimize

from .model import PYP, FLModel


class AbstractExtrapolation(ABC):
    """
    Abstract base class for extrapolation models.

    Defines the interface for models that can be trained on empirical data
    and used to predict scaling behavior at new sample sizes.
    """
    @abstractmethod
    def __init__(self, training_data: pd.DataFrame) -> None:
        """
        Initialize the extrapolation model.

        Args:
            training_data: DataFrame containing training samples
        """
        pass
    
    @classmethod
    @abstractmethod
    def make_loss_fun(cls, dd: pd.DataFrame) -> Callable:
        """
        Loss function for model optimization.

        Args:
            dd: DataFrame containing training data

        Returns:
            Callable: Loss function for optimization
        """
        pass
      
    @abstractmethod
    def train(self) -> None:
        """Train the model on the provided training data."""
        pass
    
    @abstractmethod
    def test(self, n: Union[int, ndarray]):
        """
        Make predictions for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted values for the given sample sizes
        """
        pass


class PYPExtrapolation(AbstractExtrapolation):
    """
    Extrapolation model based on Pitman-Yor Process.

    This model fits a Pitman-Yor Process to training data and uses it
    to predict scaling behavior at new sample sizes.
    """
    INIT_STATE = (12, 0.26)  # Initial state for optimization (h, γ)

    def __init__(self, training_data: pd.DataFrame) -> None:
        """
        Initialize PYP extrapolation model.

        Args:
            training_data: DataFrame with columns 'n' (sample sizes) and 'κ' (observations)
        """
        self.training_data = training_data
        self.train()

    @staticmethod
    def make_loss_fun(dd: pd.DataFrame) -> Callable:
        """
        Create loss function for PYP parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization
        """
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].values

        def loss_fun(x):
            expected_κ = PYP(h=x[0], γ=x[1]).correctness(n_range)
            return (np.log(n_range) * (expected_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self) -> None:
        """
        Train the model by optimizing PYP parameters.
        
        Uses Nelder-Mead optimization to find optimal h and γ values.
        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, type(self).INIT_STATE, method='Nelder-Mead')
        self.h, self.γ = res.x
        
    def test(self, n: Union[int, ndarray]) -> ndarray:
        """
        Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values
        """
        return PYP(h=self.h, γ=self.γ).correctness(n)


class FLExtrapolation(AbstractExtrapolation):
    """
    Extrapolation model based on baseline entropy model.

    This model provides a simpler alternative to PYP extrapolation,
    using only entropy-based calculations.
    """
    INIT_STATE = 12  # Initial entropy value for optimization

    def __init__(self, training_data):
        """
        Initialize FL extrapolation model.

        Args:
            training_data: DataFrame with columns 'n' and 'κ'
        """
        self.training_data = training_data
        self.train()

    @classmethod
    def make_loss_fun(cls, dd):
        """
        Loss function for entropy parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization
        """
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].to_numpy()

        def loss_fun(x):
            expected_κ = FLModel(h=x).correctness(n_range)
            return (np.log(n_range) * (expected_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self):
        """
        Train the model by optimizing the entropy parameter.
        
        Uses Nelder-Mead optimization to find optimal entropy value.
        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, type(self).INIT_STATE, method='Nelder-Mead')
        self.h = res.x

    def test(self, n):
        """
        Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values
        """
        return FLModel(h=self.h).correctness(n)


class ExpDecayExtrapolation(AbstractExtrapolation):
    """
    Exponential decay-based extrapolation model.

    This model fits an exponential decay curve to the training data
    for predicting scaling behavior.
    """
    def __init__(self, training_data):
        """
        Initialize exponential decay model.

        Args:
            training_data: DataFrame with columns 'n' and 'κ'
        """
        self.training_data = training_data
        self.train()
      
    @staticmethod
    def correctness(a, b, n):
        """
        Compute correctness using exponential decay formula.

        Args:
            a: Amplitude parameter
            b: Decay rate parameter
            n: Sample size(s)

        Returns:
            ndarray: Correctness values
        """
        return a * np.exp(-b * np.sqrt(n)) + (1-a * np.exp(-b))

    @classmethod
    def make_loss_fun(cls, dd):
        """
        Loss function for exponential decay parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization
        """
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].to_numpy()

        def loss_fun(x):
            est_κ = cls.correctness(x[0], x[1], n_range)
            return (np.log(n_range) * (est_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self):
        """
        Train the model by optimizing exponential decay parameters.
        
        Uses Nelder-Mead optimization to find optimal a and b values.
        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, (1, 1), method='Nelder-Mead')
        self.a, self.b = res.x

    def test(self, n):
        """
        Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values, clipped to [0,1]
        """
        return np.clip(type(self).correctness(self.a, self.b, n), 0, 1)


class PolynomialExtrapolation(AbstractExtrapolation):
    """
    Polynomial fit-based extrapolation model.

    This model fits a third-degree polynomial to the log-transformed data
    for predicting scaling behavior.
    """
    def __init__(self, training_data):
        """
        Initialize polynomial extrapolation model.

        Args:
            training_data: DataFrame with columns 'n' and 'κ'
        """
        self.training_data = training_data
        self.train()

    @staticmethod
    def correctness(a, b, c, n):
        """
        Compute correctness using polynomial formula.

        Args:
            a: Cubic coefficient
            b: Quadratic coefficient
            c: Linear coefficient
            n: Sample size(s)

        Returns:
            ndarray: Correctness values
        """
        return a * np.log10(n)**3 + b * np.log10(n)**2 + c * np.log10(n) + 1

    @classmethod
    def make_loss_fun(cls, dd):
        """
        Loss function for polynomial parameter optimization.

        Args:
            dd: DataFrame with columns 'n' and 'κ'

        Returns:
            Callable: Loss function for optimization
        """
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].to_numpy()

        def loss_fun(x):
            est_κ = cls.correctness(x[0], x[1], x[2], n_range)
            return (np.log(n_range) * (est_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self):
        """
        Train the model by optimizing polynomial coefficients.
        
        Uses Nelder-Mead optimization to find optimal polynomial coefficients.
        """
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, (0, -1, -1), method='Nelder-Mead')
        self.a, self.b, self.c = res.x

    def test(self, n):
        """
        Predict correctness values for new sample sizes.

        Args:
            n: Sample size or array of sample sizes

        Returns:
            ndarray: Predicted correctness values, clipped to [0,1]
        """
        return np.clip(type(self).correctness(self.a, self.b, self.c, n), 0, 1)
