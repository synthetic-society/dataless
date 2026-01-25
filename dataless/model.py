from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import integrate
from scipy.special import beta, digamma, gammaln, polygamma
from scipy.stats import binom

from .exceptions import ParameterError

UnionInt = npt.NDArray[np.int_] | int


def _log_gamma_ratio(z, m, *, shift_target=2.0, m_taylor=1e-3):
    """Robust log Γ(z+m) - log Γ(z) for z>0 and z+m>0."""
    z = np.asarray(z, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    zp = z + m
    if np.any(z <= 0) or np.any(zp <= 0):
        raise ValueError("Require z>0 and z+m>0 for _log_gamma_ratio.")

    # shift amount k so that min(z+k, z+m+k) >= shift_target
    min_arg = np.minimum(z, zp)
    k = np.maximum(0.0, np.ceil(shift_target - min_arg)).astype(np.int64)
    zk = z + k
    zpk = zp + k  # = z+m+k

    # core term evaluated at safe arguments
    # Use Taylor when |m| is small, OR when z is large (Taylor converges faster for large z)
    # For large z, gammaln differences lose precision due to catastrophic cancellation
    use_taylor = (np.abs(m) <= m_taylor) | (zk >= 1e6)
    core = np.where(
        use_taylor,
        m * digamma(zk) + 0.5 * (m * m) * polygamma(1, zk) + (m * m * m) * (1.0 / 6.0) * polygamma(2, zk),
        gammaln(zpk) - gammaln(zk),
    )

    # adjustment sum: sum_{i=0}^{k-1} log((z+i)/(z+m+i))
    kmax = int(np.max(k)) if np.size(k) else 0
    if kmax == 0:
        return core

    adj = np.zeros_like(core)
    # accumulate in a vectorized way; k is usually tiny (<=2) for problematic regimes
    for i in range(kmax):
        mask = k > i
        if np.any(mask):
            adj = np.where(mask, adj + np.log(z + i) - np.log(zp + i), adj)

    return core + adj


def pyp_entropy(d: float, alpha: float) -> float:
    """Calculate the expected entropy of a Pitman-Yor process.

    Args:
        d: The discount parameter (0 <= d < 1)
        alpha: The concentration parameter (alpha > -d)

    Returns:
        float: The expected entropy value

    """
    return digamma(alpha + 1) - digamma(1 - d)


def pyp_uniqueness(d, alpha, n):
    """Stable version of pyp_uniqueness for arbitrary n (scalar or array)."""
    n_arr = np.asarray(n, dtype=np.float64)
    d = np.float64(d)
    alpha = np.float64(alpha)

    out = np.ones_like(n_arr, dtype=np.float64)
    mask = n_arr > 1
    if not np.any(mask):
        return out

    nm = n_arr[mask]

    if alpha == 0.0:
        logU = _log_gamma_ratio(n_arr[mask], d - 1.0) - gammaln(d)
        out[mask] = 1.0 + np.expm1(logU)
        return np.clip(out, 0.0, 1.0)

    if d == 0.0:
        out[mask] = alpha / (nm + alpha - 1.0)
        return np.clip(out, 0.0, 1.0)

    logU = _log_gamma_ratio(alpha + d, 1.0 - d) + _log_gamma_ratio(nm + alpha, d - 1.0)
    out[mask] = np.exp(logU)
    return np.clip(out, 0.0, 1.0)


def _stable_R_minus_alpha(logR, alpha):
    R_is_finite = np.isfinite(logR)
    if not np.all(R_is_finite):
        return np.exp(logR) - alpha

    if alpha == 0.0:
        return np.exp(logR)

    if alpha < 0.0:
        return np.exp(np.logaddexp(logR, np.log(-alpha)))

    loga = np.log(alpha)
    t = loga - logR  # log(alpha/R)
    return np.exp(logR) * (-np.expm1(t))


def pyp_correctness(d, alpha, n):
    n_arr = np.asarray(n, dtype=np.float64)
    d = float(d)
    alpha = float(alpha)

    out = np.ones_like(n_arr, dtype=np.float64)
    mask = n_arr > 1
    if not np.any(mask):
        return out

    nm = n_arr[mask]

    if d == 0.0:
        if alpha <= 1e-300:
            out[mask] = 0.0  # Degenerate case: both d≈0 and α≈0
        else:
            dpsi = digamma(nm + alpha) - digamma(alpha)
            out[mask] = (alpha / nm) * dpsi
        return np.clip(out, 0.0, 1.0)

    if alpha == 0.0:
        logC = _log_gamma_ratio(nm + 1.0, d - 1.0) - gammaln(d + 1.0)
        out[mask] = 1.0 + np.expm1(logC)
        return np.clip(out, 0.0, 1.0)

    logR = _log_gamma_ratio(alpha + d, 1.0 - d) + _log_gamma_ratio(nm + alpha, d)

    if abs(d) <= 1e-6 and alpha > 1e-10:
        dpsi = digamma(nm + alpha) - digamma(alpha)
        k0 = (alpha / nm) * dpsi
        out[mask] = np.clip(k0, 0.0, 1.0)
        return out

    nom = _stable_R_minus_alpha(logR, alpha)
    k = nom / (nm * d)

    out[mask] = np.clip(k, 0.0, 1.0)
    return out


@np.vectorize
def invdigamma(y: float, num_iter=5) -> float:
    """Compute the inverse digamma function using Newton's method.

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
        x_new = x_old = np.exp(y) + 0.5
    else:
        gamma_val = -digamma(1)
        x_new = x_old = -(1 / (y + gamma_val))

    # do Newton update here
    for i in range(num_iter):
        numerator = digamma(x_old) - y
        denumerator = polygamma(1, x_old)
        x_new = x_old - (numerator / denumerator)
        x_old = x_new

    return x_new


def multiplicities_from_sample(sample: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
    """Convert an array of observations to multiplicities.

    Args:
        sample: Array of observations

    Returns:
        Tuple[ndarray, ndarray]: (multiplicities, counts)

    """
    _, freqs = np.unique(sample, return_counts=True)
    return multiplicities_from_freqs(freqs)


def multiplicities_from_freqs(freqs: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
    """Convert frequency counts to multiplicities.

    Args:
        freqs: Array of frequency counts

    Returns:
        Tuple[ndarray, ndarray]: (multiplicities, counts)

    """
    icts, mm = np.unique(freqs, return_counts=True)
    return (mm, icts)


def freqs_from_multiplicities(mm: npt.NDArray, icts: npt.NDArray) -> npt.NDArray:
    """Convert multiplicities back to frequencies.

    Args:
        mm: Multiplicities array
        icts: Counts array

    Returns:
        ndarray: Array of frequencies

    """
    return np.repeat(icts, mm)


@np.vectorize
def _scalar_kanon_violations(n: int, k: int, d: float, alpha: float):
    """Calculate k-anonymity violations in a Pitman-Yor process sample.

    Args:
        n: Sample size
        k: Anonymity parameter
        d: Discount parameter
        alpha: Concentration parameter

    Returns:
        float: Probability of k-anonymity violation

    """

    def q(x):
        return binom.sf(k - 2, n - 1, x) * x ** (-d) * (1 - x) ** (alpha + d - 1)

    def q_transform(u):
        rv = q(np.exp(u)) * np.exp(u)
        if np.isnan(rv):
            return 0.0
        return rv

    rv = 1 - integrate.quad(q, 0, 1)[0] / beta(1 - d, alpha + d)

    if rv > 0.99 or rv < 0.01:
        rv = 1 - integrate.quad(q_transform, -np.inf, 0)[0] / beta(1 - d, alpha + d)

    return rv


class AbstractModel(ABC):
    """Abstract base class for statistical models.

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
    """Pitman-Yor Process model implementation.

    The Pitman-Yor Process is a generalization of the Dirichlet Process,
    providing more flexible modeling of power-law behavior in the tail of
    the distribution.

    Args:
        d: Discount parameter (0 <= d < 1)
        alpha: Concentration parameter (alpha > -d)
        h: Entropy parameter (in nats)
        gamma: Power-law exponent

    Note:
        Must provide either (d, alpha) or (h, gamma) pair.

    Raises:
        ParameterError: If parameters are invalid or constraints violated.

    """

    def __init__(
        self, d: float | None = None, alpha: float | None = None, h: float | None = None, gamma: float | None = None
    ) -> None:
        """Initialize a Pitman-Yor Process model.

        Args:
            d: Discount parameter (0 <= d < 1)
            alpha: Concentration parameter (alpha > -d)
            h: Entropy parameter (in nats)
            gamma: Power-law exponent

        Raises:
            ParameterError: If parameters are invalid

        """
        if d is not None and alpha is not None:
            if not (0 <= d < 1):
                raise ParameterError(f"Discount parameter d must be in [0, 1), got {d}")
            if not (alpha > -d):
                raise ParameterError(f"Concentration alpha must be > -d, got alpha={alpha}, d={d}")
            self.d: float = d
            self.alpha: float = alpha
        elif h is not None and gamma is not None:
            if h <= 0:
                raise ParameterError(f"Entropy h must be positive, got {h}")
            if not (0 <= gamma <= 1):
                raise ParameterError(f"Power-law exponent gamma must be in [0, 1], got {gamma}")
            self.d: float = 1 - invdigamma(digamma(1) - h * gamma)
            self.alpha: float = invdigamma(h * (1 - gamma) + digamma(1)) - 1
        else:
            raise ParameterError("PYP() must be instantiated either with d and alpha or with h and gamma.")

    @property
    def h(self) -> float:
        """Expected entropy of the process."""
        return pyp_entropy(self.d, self.alpha)

    @property
    def gamma(self) -> float:
        """Power-law exponent of the process."""
        return (digamma(1) - digamma(1 - self.d)) / self.h

    def uniqueness(self, n):
        """Calculate expected uniqueness for sample size n."""
        return pyp_uniqueness(self.d, self.alpha, n)

    def correctness(self, n):
        """Calculate expected correctness for sample size n."""
        return pyp_correctness(self.d, self.alpha, n)

    def kanon_violations(self, n, k):
        """Calculate expected k-anonymity violations for sample size n."""
        return _scalar_kanon_violations(n, k, self.d, self.alpha)


class FLModel(AbstractModel):
    """Baseline entropy model for comparison purposes.

    This model provides a simpler alternative to the PYP model,
    based solely on entropy calculations.
    """

    def __init__(self, h: float):
        """Initialize the baseline entropy model.

        Args:
            h: Entropy parameter (clipped to [0, 100])

        """
        self.h = np.clip(h, 0, 100)

    def uniqueness(self, n):
        """Calculate expected uniqueness for sample size n."""
        # (1 - 2^(-h))^(n-1) = exp((n-1) * log(1 - 2^(-h)))
        x = 2.0 ** (-self.h)
        return np.clip(np.exp(((n - 1) * np.log1p(-x))), 0.0, 1.0)

    def correctness(self, n):
        """Calculate expected correctness for sample size n."""
        if self.h == 0:
            return np.where(np.asarray(n) >= 1, 0.0, np.nan)

        # Numerically stable computation:
        x = 2.0 ** (-self.h)
        stable_term = -np.expm1(n * np.log1p(-x))

        return np.clip(2.0**self.h / n * stable_term, 0.0, 1.0)

    def kanon_violations(self, n, k):
        """Not implemented for baseline model."""
        raise NotImplementedError
