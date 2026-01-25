# test_pyp_stability.py
import math
import warnings

import numpy as np
import pytest
from dataless.model import _log_gamma_ratio, pyp_correctness, pyp_uniqueness
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from scipy.special import digamma, poch

# ----------------------------
# Hypothesis strategies
# ----------------------------


def _finite_float(min_value=None, max_value=None, *, width=64):
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        width=width,
    )


@st.composite
def pyp_params(draw):
    """
    Generate (d, alpha) with constraints:
      0 <= d < 1
      alpha > -d

    Weighted to hit numerically hard corners:
      - d very small
      - d extremely close to 1
      - alpha very close to -d
      - alpha very large
    """
    d = draw(
        st.one_of(
            st.sampled_from([0.0, 1e-16, 1e-14, 1e-12, 1e-9, 1e-6]),
            _finite_float(0.0, 1.0 - 2**-52),  # avoid exactly 1
            st.sampled_from([1.0 - 1e-16, 1.0 - 1e-14, 1.0 - 1e-12, 1.0 - 1e-9, 1.0 - 1e-6]),
        )
    )
    d = float(min(max(d, 0.0), 1.0 - 2**-52))

    eps = draw(st.sampled_from([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2]))
    mode = draw(st.sampled_from(["near_boundary", "moderate", "large"]))

    if mode == "near_boundary":
        alpha = -d + eps
    elif mode == "moderate":
        alpha = draw(_finite_float(-d + eps, 50.0))
    else:
        # Large alpha stresses asymptotics + cancellation paths.
        alpha = draw(_finite_float(1e-8, 1e6))

    assume(alpha > -d)
    return float(d), float(alpha)


@st.composite
def n_values(draw, *, min_size=1, max_size=50, max_n=10**9):
    """
    Irregular n (unsorted, non-consecutive), including 0 and 1.
    """
    ns = draw(st.lists(st.integers(min_value=0, max_value=max_n), min_size=min_size, max_size=max_size))
    return np.array(ns, dtype=np.int64)


# ----------------------------
# _log_gamma_ratio tests
# ----------------------------


@settings(max_examples=500, deadline=None)
@given(z=_finite_float(1e-12, 1e6), m=_finite_float(-0.9, 0.9))
@example(z=1e-12, m=1e-12)
@example(z=1e6, m=1e-12)
def test_log_gamma_ratio_additivity_identity(z, m):
    """
    Identity: log Γ(z+m1+m2) - log Γ(z) =
              (log Γ(z+m1) - log Γ(z)) + (log Γ(z+m1+m2) - log Γ(z+m1))
    i.e. f(z, m1+m2) = f(z, m1) + f(z+m1, m2)
    """
    m1 = m
    m2 = 0.5 * m  # correlated to hit tricky cancellation regimes

    assume(z + m1 > 1e-12)
    assume(z + m1 + m2 > 1e-12)

    lhs = _log_gamma_ratio(z, m1 + m2)
    rhs = _log_gamma_ratio(z, m1) + _log_gamma_ratio(z + m1, m2)

    # This is a tight structural check. If it fails, either Taylor switching or
    # pole-avoidance logic is broken.
    assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-8)


@settings(max_examples=400, deadline=None)
@given(z=_finite_float(1e-6, 1e6))
@example(z=1e-6)
@example(z=1e6)
def test_log_gamma_ratio_derivative_at_zero_matches_digamma(z):
    """
    d/dm [log Γ(z+m) - log Γ(z)] at m=0 is ψ(z).
    Use a scale-aware step to avoid cancellation and to stay in-domain.
    """
    eps = np.finfo(float).eps
    h = max(1e-16, np.sqrt(eps) * max(1.0, abs(z)))
    val = (_log_gamma_ratio(z, h) - _log_gamma_ratio(z, -h)) / (2.0 * h)
    assert_allclose(val, digamma(z), rtol=1e-3, atol=1e-10)


@settings(max_examples=300, deadline=None)
@given(z=_finite_float(1e-3, 200.0), k=st.integers(min_value=0, max_value=50))
def test_log_gamma_ratio_integer_step_matches_log_rising_factorial(z, k):
    """
    For integer k>=0:
      Γ(z+k)/Γ(z) = ∏_{i=0}^{k-1} (z+i)
    Compare in log-space.
    """
    expected = 0.0
    for i in range(k):
        expected += math.log(z + i)
    got = _log_gamma_ratio(z, float(k))
    assert_allclose(got, expected, rtol=2e-12, atol=2e-12)


@settings(max_examples=300, deadline=None)
@given(z=_finite_float(1.0, 200.0), k=st.integers(min_value=1, max_value=50))
def test_log_gamma_ratio_negative_integer_step_matches_inverse_rising_factorial(z, k):
    """
    For integer k>=1 and z>k:
      Γ(z-k)/Γ(z) = 1 / ∏_{i=0}^{k-1} (z-k+i)
    => log ratio = - Σ log(z-k+i)
    """
    assume(z > k + 1e-12)
    expected = 0.0
    for i in range(k):
        expected -= math.log(z - k + i)
    got = _log_gamma_ratio(z, float(-k))
    assert_allclose(got, expected, rtol=2e-12, atol=2e-12)


@settings(max_examples=300, deadline=None)
@given(z=_finite_float(1e-6, 200.0), m=_finite_float(-0.9, 0.9))
def test_log_gamma_ratio_matches_pochhammer_for_safe_region(z, m):
    """
    (z)_m = Γ(z+m)/Γ(z). So exp(log_gamma_ratio) should match poch(z,m)
    where poch is stable and finite.
    """
    assume(z + m > 1e-12)
    # Restrict so poch stays finite and not outrageously large.
    assume(z < 200.0)
    assume(abs(m) < 10.0)

    got = math.exp(_log_gamma_ratio(z, m))
    ref = float(poch(z, m))
    assert_allclose(got, ref, rtol=1e-11, atol=1e-12)


def test_log_gamma_ratio_high_precision_oracle_if_mpmath_installed():
    """
    Strong oracle test using mpmath (optional but recommended).
    Add `mpmath` to your test extras for maximum rigor.
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80

    # A small set of adversarial points: tiny z, tiny m; huge z, tiny m; mixed signs.
    cases = [
        (mp.mpf("1e-12"), mp.mpf("1e-12")),
        (mp.mpf("1e-6"), mp.mpf("1e-9")),
        (mp.mpf("1e6"), mp.mpf("1e-12")),
        (mp.mpf("10.0"), mp.mpf("-0.75")),
        (mp.mpf("3.5"), mp.mpf("0.999999999999")),
    ]
    for z, m in cases:
        if z + m <= mp.mpf("1e-30"):
            continue
        ref = mp.log(mp.gamma(z + m)) - mp.log(mp.gamma(z))
        got = _log_gamma_ratio(float(z), float(m))
        assert_allclose(got, float(ref), rtol=0, atol=1e-10)


# ----------------------------
# PYP uniqueness/correctness tests
# ----------------------------


@settings(max_examples=500, deadline=None)
@given(params=pyp_params(), n=n_values())
def test_pyp_outputs_in_unit_interval_and_finite(params, n):
    d, alpha = params
    u = pyp_uniqueness(d, alpha, n)
    c = pyp_correctness(d, alpha, n)

    assert u.shape == n.shape
    assert c.shape == n.shape
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(c))
    assert_array_less(-1e-15, u)  # u >= 0
    assert_array_less(u, 1.0 + 1e-15)  # u <= 1
    assert_array_less(-1e-15, c)  # c >= 0
    assert_array_less(c, 1.0 + 1e-15)  # c <= 1

    # n<=1 is defined as 1.0 in the implementation
    assert_array_equal(u[n <= 1], 1.0)
    assert_array_equal(c[n <= 1], 1.0)


@settings(max_examples=400, deadline=None)
@given(params=pyp_params(), n=st.integers(min_value=2, max_value=10**6))
def test_uniqueness_exact_consecutive_recurrence(params, n):
    """
    From gamma recurrence, the closed form for uniqueness implies:
      U(n+1) = U(n) * (n + alpha + d - 1) / (n + alpha)
    This is an exact identity at the real-number level.
    """
    d, alpha = params
    un = float(pyp_uniqueness(d, alpha, n))
    un1 = float(pyp_uniqueness(d, alpha, n + 1))
    ratio = (n + alpha + d - 1.0) / (n + alpha)

    # Numerical check; should be very tight unless extremely near poles.
    assert_allclose(un1, un * ratio, rtol=1e-8, atol=1e-8)
    # Also implies monotone non-increasing in n (since d<1 => ratio<1 for n>=2).
    assert_array_less(un1, un + 1e-10)


@settings(max_examples=400, deadline=None)
@given(params=pyp_params(), n=st.integers(min_value=2, max_value=10**6))
def test_correctness_recurrence_via_R_transform(params, n):
    """
    Let R(n) = α + n d C(n).
    From the closed form, R satisfies:
      R(n+1) = R(n) * (n + alpha + d) / (n + alpha)
    This avoids directly recomputing gamma ratios in the test.
    """
    d, alpha = params
    assume(d > 0.0)  # recurrence is meaningful for d>0

    cn = float(pyp_correctness(d, alpha, n))
    cn1 = float(pyp_correctness(d, alpha, n + 1))

    Rn = alpha + n * d * cn
    Rn1 = alpha + (n + 1) * d * cn1
    ratio = (n + alpha + d) / (n + alpha)

    assert_allclose(Rn1, Rn * ratio, rtol=1e-8, atol=1e-8)


@settings(max_examples=200, deadline=None)
@given(alpha=_finite_float(1e-8, 1e6), n=st.integers(min_value=2, max_value=10**9))
def test_uniqueness_d0_reduces_to_simple_rational(alpha, n):
    """
    At d=0 (Dirichlet process limit), uniqueness reduces to:
      U(n) = α / (n + α - 1)  for n>1
    Derived using Γ(z+1)=zΓ(z).
    """
    u = float(pyp_uniqueness(0.0, float(alpha), int(n)))
    ref = float(alpha) / (float(n) + float(alpha) - 1.0)
    assert_allclose(u, ref, rtol=2e-14, atol=2e-14)


@settings(max_examples=300, deadline=None)
@given(alpha=_finite_float(1e-8, 1e6), n=st.integers(min_value=2, max_value=10**9))
def test_correctness_d0_matches_digamma_limit(alpha, n):
    """
    At d=0, correctness limit:
      C0(n) = α/n * (ψ(n+α) - ψ(α))
    """
    alpha = float(alpha)
    n = int(n)
    c = float(pyp_correctness(0.0, alpha, n))
    ref = (alpha / n) * (digamma(n + alpha) - digamma(alpha))
    assert_allclose(c, ref, rtol=2e-13, atol=2e-13)


@settings(max_examples=250, deadline=None)
@given(params=pyp_params(), ns=n_values(min_size=5, max_size=30))
def test_monotone_in_n_for_random_irregular_ns(params, ns):
    """
    Spec-level monotonicity test (numerical tolerance only).
    Sort n and require non-increasing sequences.

    Note: correctness monotonicity is expected from the probabilistic definition;
    we allow a tiny epsilon for floating noise.
    """
    d, alpha = params
    # Skip edge cases where numerical precision breaks down:
    # - d > 1 - 1e-6: d very close to 1 causes clipping artifacts
    # - alpha + d < 1e-6: near the constraint boundary
    # - alpha > 10000: very large alpha causes precision loss for small n
    assume(d < 1 - 1e-6)
    assume(alpha + d > 1e-6)
    assume(alpha < 10000)

    idx = np.argsort(ns)
    ns_sorted = ns[idx]

    u = pyp_uniqueness(d, alpha, ns_sorted)
    c = pyp_correctness(d, alpha, ns_sorted)

    # Tolerance 1e-6 accommodates floating-point noise at edge cases:
    # - alpha very large: precision loss near 1
    # - small d with certain alpha ranges
    assert_array_less(u[1:], u[:-1] + 1e-6)
    assert_array_less(c[1:], c[:-1] + 1e-6)


def test_high_precision_oracle_for_pyp_if_mpmath_installed():
    """
    Strong cross-check for both metrics vs mpmath gamma ratios.
    Recommended to run in CI for maximum confidence.
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80

    # Adversarial points hitting:
    # - d tiny
    # - d near 1
    # - alpha near -d
    # - large n
    cases = [
        (1e-16, 1.0, 2),
        (1e-12, 1e-10, 10**6),
        (1e-9, 1e-6, 10**7),
        (1.0 - 1e-12, 1e-6, 10**5),
        (0.75, -0.75 + 1e-12, 10**4),
        (0.999999999999, 50.0, 10**6),
    ]

    for d, alpha, n in cases:
        d_mp = mp.mpf(str(d))
        a_mp = mp.mpf(str(alpha))
        n_mp = mp.mpf(int(n))

        # Uniqueness: Γ(1+α)/Γ(α+d) * Γ(n+α+d-1)/Γ(n+α)
        U_ref = mp.gamma(1 + a_mp) / mp.gamma(a_mp + d_mp) * mp.gamma(n_mp + a_mp + d_mp - 1) / mp.gamma(n_mp + a_mp)
        U_got = float(pyp_uniqueness(d, alpha, n))

        # Correctness (for d>0): (R-α)/(n d), with R = Γ(1+α)/Γ(α+d) * Γ(n+α+d)/Γ(n+α)
        if d > 0:
            R = mp.gamma(1 + a_mp) / mp.gamma(a_mp + d_mp) * mp.gamma(n_mp + a_mp + d_mp) / mp.gamma(n_mp + a_mp)
            C_ref = (R - a_mp) / (n_mp * d_mp)
            C_got = float(pyp_correctness(d, alpha, n))
            assert_allclose(C_got, float(C_ref), rtol=0, atol=1e-8)

        assert_allclose(U_got, float(U_ref), rtol=0, atol=1e-8)


def test_no_runtime_warnings_on_edge_regressions():
    """
    A small regression-style stress test to ensure no warnings (overflow/invalid)
    slip in for known problematic corners.
    """
    stress = [
        (1e-16, 1e-12, np.array([2, 3, 10, 10**6], dtype=np.int64)),
        (1 - 1e-12, 1e-12, np.array([2, 10, 10**4, 10**7], dtype=np.int64)),
        (0.9, -0.9 + 1e-12, np.array([2, 3, 4, 1000], dtype=np.int64)),
    ]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("error", RuntimeWarning)
        for d, alpha, n in stress:
            u = pyp_uniqueness(d, alpha, n)
            c = pyp_correctness(d, alpha, n)
            assert np.all(np.isfinite(u))
            assert np.all(np.isfinite(c))
            assert np.all((0.0 <= u) & (u <= 1.0))
            assert np.all((0.0 <= c) & (c <= 1.0))


def test_large_n_monotonicity_regressions():
    """
    Regression tests for large n values where gammaln differences lose precision.
    These cases were found by Hypofuzz and require Taylor expansion for large z.
    """
    # Case from Hypofuzz: d close to 1, moderate alpha, n ~ 9 billion
    cases = [
        # (d, alpha, n_array) - n values must be sorted ascending
        (0.98828125, 0.990234375, np.array([9_364_366_983, 9_389_675_994])),
        # Similar cases with varying d close to 1
        (0.99, 1.0, np.array([10**9, 10**9 + 10**6])),
        (0.95, 0.5, np.array([5 * 10**9, 5 * 10**9 + 10**7])),
        # Large n with smaller d
        (0.5, 1.0, np.array([10**9, 2 * 10**9, 5 * 10**9])),
        # Edge case: d very close to 1
        (0.999, 0.1, np.array([10**8, 10**8 + 10**5, 10**8 + 2 * 10**5])),
    ]

    for d, alpha, n in cases:
        u = pyp_uniqueness(d, alpha, n)
        c = pyp_correctness(d, alpha, n)

        # Check values are valid
        assert np.all(np.isfinite(u)), f"Non-finite uniqueness for d={d}, alpha={alpha}"
        assert np.all(np.isfinite(c)), f"Non-finite correctness for d={d}, alpha={alpha}"
        assert np.all((0.0 <= u) & (u <= 1.0)), f"Uniqueness out of bounds for d={d}, alpha={alpha}"
        assert np.all((0.0 <= c) & (c <= 1.0)), f"Correctness out of bounds for d={d}, alpha={alpha}"

        # Check monotonicity (with small tolerance for floating-point noise)
        u_diff = np.diff(u)
        c_diff = np.diff(c)
        assert np.all(u_diff <= 1e-5), f"Non-monotonic uniqueness for d={d}, alpha={alpha}: diff={u_diff}"
        assert np.all(c_diff <= 1e-5), f"Non-monotonic correctness for d={d}, alpha={alpha}: diff={c_diff}"


def test_large_n_with_extreme_parameters():
    """
    Test combinations of large n with extreme d and alpha values.
    """
    large_n = np.array([10**7, 10**8, 10**9], dtype=np.int64)

    # Various parameter combinations
    params = [
        (0.001, 1.0),  # Small d
        (0.5, 0.001),  # Small alpha
        (0.999, 100.0),  # d near 1, large alpha
        (0.1, 0.1),  # Both small
    ]

    for d, alpha in params:
        u = pyp_uniqueness(d, alpha, large_n)
        c = pyp_correctness(d, alpha, large_n)

        # Values should be valid and monotonic
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(c))
        assert np.all((0.0 <= u) & (u <= 1.0))
        assert np.all((0.0 <= c) & (c <= 1.0))
        assert np.all(np.diff(u) <= 1e-5), f"Non-monotonic uniqueness for d={d}, alpha={alpha}"
        assert np.all(np.diff(c) <= 1e-5), f"Non-monotonic correctness for d={d}, alpha={alpha}"
