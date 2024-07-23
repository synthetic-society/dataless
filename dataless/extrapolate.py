from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.optimize import minimize

from .model import PYP, FLModel


class AbstractExtrapolation(ABC):
    @abstractmethod
    def __init__(self, training_data: pd.DataFrame) -> None:
        pass
    
    @classmethod
    @abstractmethod
    def make_loss_fun(cls, dd: pd.DataFrame) -> Callable:
        pass
      
    @abstractmethod
    def train(self) -> None:
        pass
    
    @abstractmethod
    def test(self, n: Union[int, ndarray]):
        pass



class EpycExtrapolation(AbstractExtrapolation):
    INIT_STATE = (12, 0.26)  # arbitrary initial state

    def __init__(self, training_data: pd.DataFrame) -> None:
        self.training_data = training_data
        self.train()

    @staticmethod
    def make_loss_fun(dd: pd.DataFrame) -> Callable:
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].values

        def loss_fun(x):
            expected_κ = PYP(h=x[0], γ=x[1]).correctness(n_range)
            return (np.log(n_range) * (expected_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self) -> None:
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, type(self).INIT_STATE, method='Nelder-Mead')
        self.h, self.γ = res.x
        
    def test(self, n: Union[int, ndarray]) -> ndarray:
        return PYP(h=self.h, γ=self.γ).correctness(n)


class FLExtrapolation(AbstractExtrapolation):
    INIT_STATE = 12

    def __init__(self, training_data):
        self.training_data = training_data
        self.train()

    @classmethod
    def make_loss_fun(cls, dd):
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].to_numpy()

        def loss_fun(x):
            expected_κ = FLModel(h=x).correctness(n_range)
            return (np.log(n_range) * (expected_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self):
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, type(self).INIT_STATE, method='Nelder-Mead')
        self.h = res.x

    def test(self, n):
        return FLModel(h=self.h).correctness(n)


class ExpDecayExtrapolation(AbstractExtrapolation):
    def __init__(self, training_data):
        self.training_data = training_data
        self.train()
      
    @staticmethod
    def correctness(a, b, n):
        return a * np.exp(-b * np.sqrt(n)) + (1-a * np.exp(-b))

    @classmethod
    def make_loss_fun(cls, dd):
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].to_numpy()

        def loss_fun(x):
            est_κ = cls.correctness(x[0], x[1], n_range)
            return (np.log(n_range) * (est_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self):
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, (1, 1), method='Nelder-Mead')
        self.a, self.b = res.x

    def test(self, n):
        return np.clip(type(self).correctness(self.a, self.b, n), 0, 1)


class PolynomialExtrapolation(AbstractExtrapolation):
    def __init__(self, training_data):
        self.training_data = training_data
        self.train()

    @staticmethod
    def correctness(a, b, c, n):
        return a * np.log10(n)**3 + b * np.log10(n)**2 + c * np.log10(n) + 1

    @classmethod
    def make_loss_fun(cls, dd):
        n_range = dd.n.to_numpy()
        emp_κ = dd['κ'].to_numpy()

        def loss_fun(x):
            est_κ = cls.correctness(x[0], x[1], x[2], n_range)
            return (np.log(n_range) * (est_κ - emp_κ)**2).mean()

        return loss_fun

    def train(self):
        loss_function = type(self).make_loss_fun(self.training_data)
        res = minimize(loss_function, (0, -1, -1), method='Nelder-Mead')
        self.a, self.b, self.c = res.x

    def test(self, n):
        return np.clip(type(self).correctness(self.a, self.b, self.c, n), 0, 1)
