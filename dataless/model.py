from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import integrate
from scipy.special import beta, digamma, gammaln, polygamma
from scipy.stats import binom


UnionInt = Union[npt.NDArray[np.int_], int]


def pyp_entropy(d: float, α: float) -> float:
    """
    Calculate the expected entropy of a Pitman-Yor process.

    Args:
        d: The discount parameter (0 ≤ d < 1)
        α: The concentration parameter (α > -d)

    Returns:
        float: The expected entropy value
    """
    return digamma(α + 1) - digamma(1 - d)

def pyp_uniqueness(d, α, n: UnionInt):
    """
    Calculate the expected uniqueness in a Pitman-Yor process sample.

    Args:
        d: The discount parameter (0 ≤ d < 1)
        α: The concentration parameter (α > -d)
        n: Sample size or array of sample sizes

    Returns:
        float or ndarray: Expected uniqueness value(s)
    """
    rv = np.exp(gammaln(1 + α) - gammaln(d + α) + gammaln(n + d + α - 1) - gammaln(n + α))
    return np.where(n <= 1, 1., rv)


def pyp_correctness(d, α, n: UnionInt):
    """
    Calculate the expected correctness in a Pitman-Yor process sample.

    Args:
        d: The discount parameter (0 ≤ d < 1)
        α: The concentration parameter (α > -d)
        n: Sample size or array of sample sizes

    Returns:
        float or ndarray: Expected correctness value(s)
    """
    d = np.asarray(d)
    rv_null_d = α/n * (digamma(n+α) - digamma(α))
    
    nom = np.exp(gammaln(1 + α) - gammaln(d + α) + gammaln(n + d + α) - gammaln(n + α)) - α

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n <= 1,
                        1.,
                        np.where(d == 0.,
                                 rv_null_d,
                                 np.divide(nom, (n * d))))

@np.vectorize
def invdigamma(y: float, num_iter=5) -> float:
    """
    Compute the inverse digamma function using Newton's method.

    Implementation based on Minka (2003) "Estimating a Dirichlet distribution",
    Appendix C.

    Args:
        y: Input value
        num_iter: Number of Newton iterations (default: 5)

    Returns:
        float: The inverse digamma value
    """
    assert num_iter > 0

    # initialisation
    if y >= -2.22:
        x_new = x_old = np.exp(y)+0.5
    else:
        gamma_val = -digamma(1)
        x_new = x_old = -(1/(y+gamma_val))

    # do Newton update here
    for i in range(num_iter):
        numerator = digamma(x_old) - y
        denumerator = polygamma(1, x_old)
        x_new = x_old - (numerator/denumerator)
        x_old = x_new

    return x_new


def multiplicities_from_sample(sample: npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Convert an array of observations to multiplicities.

    Args:
        sample: Array of observations

    Returns:
        Tuple[ndarray, ndarray]: (multiplicities, counts)
    """
    _, freqs = np.unique(sample, return_counts=True)
    return multiplicities_from_freqs(freqs)


def multiplicities_from_freqs(freqs: npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Convert frequency counts to multiplicities.

    Args:
        freqs: Array of frequency counts

    Returns:
        Tuple[ndarray, ndarray]: (multiplicities, counts)
    """
    icts, mm = np.unique(freqs, return_counts=True)
    return (mm, icts)


def freqs_from_multiplicities(mm: npt.NDArray, icts: npt.NDArray) -> npt.NDArray:
    """
    Convert multiplicities back to frequencies.

    Args:
        mm: Multiplicities array
        icts: Counts array

    Returns:
        ndarray: Array of frequencies
    """
    return np.repeat(icts, mm)

    
@np.vectorize
def _scalar_kanon_violations(n: int, k: int, d:float , α:float):
    """
    Calculate k-anonymity violations in a Pitman-Yor process sample.

    Args:
        n: Sample size
        k: Anonymity parameter
        d: Discount parameter
        α: Concentration parameter

    Returns:
        float: Probability of k-anonymity violation
    """
    def q(x):
        return binom.sf(k-2, n-1, x) * x**(- d) * (1-x)**(α + d - 1)
    
    def q_transform(u):
        rv = q(np.exp(u)) * np.exp(u)
        if np.isnan(rv): return 0.
        return rv
    
    rv = 1-integrate.quad(q, 0, 1)[0] / beta(1 - d, α + d)
    
    if rv > .99 or rv < 0.01:
        rv = 1-integrate.quad(q_transform, -np.inf, 0)[0] / beta(1 - d, α + d)
        
    return rv

  
class AbstractModel(ABC):
    """
    Abstract base class for statistical models.

    Defines the interface for models that compute correctness,
    uniqueness, and k-anonymity violations.
    """
    @abstractmethod
    def correctness(self, n: UnionInt) -> npt.NDArray[np.float64]:
        """Calculate expected correctness for sample size n."""
    
    @abstractmethod
    def uniqueness(self, n: UnionInt) -> npt.NDArray[np.float64]:
        """Calculate expected uniqueness for sample size n."""
      
    @abstractmethod
    def kanon_violations(self, n: UnionInt) -> npt.NDArray[np.float64]:
        """Calculate expected k-anonymity violations for sample size n."""
    
  
class PYP(AbstractModel):
    """
    Pitman-Yor Process model implementation.

    The Pitman-Yor Process is a generalization of the Dirichlet Process,
    providing more flexible modeling of power-law behavior in the tail of
    the distribution.
    """
    def __init__(self,
                 d: Optional[float]=None, α: Optional[float]=None,
                 h: Optional[float]=None, γ: Optional[float]=None) -> None:
        """
        Initialize a Pitman-Yor Process model.

        Args:
            d: Discount parameter (0 ≤ d < 1)
            α: Concentration parameter (α > -d)
            h: Entropy parameter (in nats)
            γ: Power-law exponent

        Note: Must provide either (d, α) or (h, γ) pair
        """
        if d is not None and α is not None:
            self.d: float = d
            self.α: float = α
        elif h is not None and γ is not None:
            self.d: float = 1 - invdigamma(digamma(1) - h * γ)
            self.α: float = invdigamma(h * (1-γ) + digamma(1)) - 1
        else:
            raise ValueError("PYP() must be instantiated either with d and α or with h and γ.")
 
    @property
    def h(self) -> float:
        """Expected entropy of the process."""
        return pyp_entropy(self.d, self.α)

    @property
    def γ(self) -> float:
        """Power-law exponent of the process."""
        return (digamma(1) - digamma(1 - self.d)) / self.h

    def uniqueness(self, n):
        """Calculate expected uniqueness for sample size n."""
        return pyp_uniqueness(self.d, self.α, n)

    def correctness(self, n):
        """Calculate expected correctness for sample size n."""
        return pyp_correctness(self.d, self.α, n)
    
    def kanon_violations(self, n, k):
        """Calculate expected k-anonymity violations for sample size n."""
        return _scalar_kanon_violations(n, k, self.d, self.α)


class FLModel(AbstractModel):
    """
    Baseline entropy model for comparison purposes.

    This model provides a simpler alternative to the PYP model,
    based solely on entropy calculations.
    """
    def __init__(self, h: float):
        """
        Initialize the baseline entropy model.

        Args:
            h: Entropy parameter (clipped to [0, 100])
        """
        self.h = np.clip(h, 0, 100)
      
    def uniqueness(self, n):
        """Calculate expected uniqueness for sample size n."""
        return (1 - 2 ** (-self.h)) ** (n - 1)

    def correctness(self, n):
        """Calculate expected correctness for sample size n."""
        return 2 ** self.h / n * (1 - (1 - 2 ** (-self.h)) ** n)
      
    def kanon_violations(self, n, k):
        """Not implemented for baseline model."""
        raise NotImplementedError
