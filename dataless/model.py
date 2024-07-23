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
    Return the expected entropy of a Pitman-Yor process given parameters
    d and α.
    """
    return digamma(α + 1) - digamma(1 - d)

def pyp_uniqueness(d, α, n: UnionInt):
    """
    Return the expected uniqueness in a sample of size n drawn from a 
    Pitman-Yor process given parameters d and α.
    """
    rv = np.exp(gammaln(1 + α) - gammaln(d + α) + gammaln(n + d + α - 1) - gammaln(n + α))
    return np.where(n <= 1, 1., rv)


def pyp_correctness(d, α, n: UnionInt):
    """
    Return the expected correctness in a sample of size n drawn from a 
    Pitman-Yor process given parameters d and α.
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
    Computes the inverse digamma function using Newton's method
    See Appendix c of Minka, T. P. (2003). Estimating a Dirichlet distribution.
    Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1 for details.
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
    """ Convert an array of observations (sample) to multiplicities. """
    _, freqs = np.unique(sample, return_counts=True)
    return multiplicities_from_freqs(freqs)


def multiplicities_from_freqs(freqs: npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
    """ Convert a list of frequency counts to multiplicities. """
    icts, mm = np.unique(freqs, return_counts=True)
    return (mm, icts)


def freqs_from_multiplicities(mm: npt.NDArray, icts: npt.NDArray) -> npt.NDArray:
    return np.repeat(icts, mm)

    
@np.vectorize
def _scalar_kanon_violations(n: int, k: int, d:float , α:float):
    # # Less accurate version:
    # q = lambda x: betainc(n-k+1, k-1, 1-x) * x**(- d) * (1-x)**(α + d - 1) / beta(1 - p.d, α + d)
    # return integrate.quad(q, 0, 1)[0] 
  
    # More accurate version:
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
    @abstractmethod
    def correctness(self, n: UnionInt) -> npt.NDArray[np.float64]:
        pass
    
    @abstractmethod
    def uniqueness(self, n: UnionInt) -> npt.NDArray[np.float64]:
        pass
      
    @abstractmethod
    def kanon_violations(self, n: UnionInt) -> npt.NDArray[np.float64]:
        pass
    
  
class PYP(AbstractModel):
    def __init__(self,
                 d: Optional[float]=None, α: Optional[float]=None,
                 h: Optional[float]=None, γ: Optional[float]=None) -> None:
        """
        Pitman-Yor process.

        Note: h is is nats, not bits
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
        return pyp_entropy(self.d, self.α)

    @property
    def γ(self) -> float:
        return (digamma(1) - digamma(1 - self.d)) / self.h

    def uniqueness(self, n):
        return pyp_uniqueness(self.d, self.α, n)

    def correctness(self, n):
        return pyp_correctness(self.d, self.α, n)
    
    def kanon_violations(self, n, k):
        return _scalar_kanon_violations(n, k, self.d, self.α)


class FLModel(AbstractModel):
    def __init__(self, h: float):
        """
        Baseline entropy model
        """
        self.h = np.clip(h, 0, 100)
      
    def uniqueness(self, n):
        return (1 - 2 ** (-self.h)) ** (n - 1)

    def correctness(self, n):
        return 2 ** self.h / n * (1 - (1 - 2 ** (-self.h)) ** n)
      
    def kanon_violations(self, n, k):
        raise NotImplementedError
