"""Split-HAlf RePeated (SHARP) significance test for comparing cross-validated
model performance.

Standard cross-validation only yields fold-level performance estimates whose
variance and between-fold correlation cannot both be estimated without
strong assumptions (Bengio & Grandvalet, 2004), which is why common tests
(e.g. the paired t-test across folds) underestimate variance and inflate
false positives. SHARP resolves this non-identifiability: in each of J
repetitions, it randomly splits the data into two disjoint halves A and B
and runs cross-validation independently within each half. The within-
repetition A/B statistics are independent (disjoint data), while
across-repetition statistics are correlated (data is reused across
repetitions), which makes the variance and correlation separately
identifiable.

Reference: Zeng, T., Li, H., Zhang, S. et al., Yeo, B.T.T. (2026),
"Widespread use of invalid statistical tests in biomedical machine
learning", bioRxiv, Methods Section 4.6.
https://doi.org/10.64898/2026.05.17.724301

======================================================================
EXPERIMENTAL -- READ THE CALIBRATION NUMBERS BELOW BEFORE RELYING ON IT
======================================================================

What is implemented
-------------------
The paper describes several ways to estimate the two nuisance parameters
(sigma2, rho) of the covariance structure above; it selects the *score
test* on the basis of a toy simulation (its Fig. S11) and uses that
variant for every result it reports. This module implements that score
test:

1. Pin the mean at the null (mu = 0) and maximize the Gaussian
   log-likelihood of D over (sigma2, rho), giving (sigma2_0, rho_0).
2. Z = D-bar / sqrt(sigma2_0 * (1 / (2J) + (J - 1) / J * rho_0)),
   which is the paper's score statistic U(0) / sqrt(I(0)) written out.
3. Two-sided p-value 2 * (1 - Phi(|Z|)); the confidence interval is
   obtained by inverting the same test (the set of mu0 that would not be
   rejected), so the interval and the p-value always agree.

Two deliberate departures from a literal transcription of the paper's
implementation, both of which leave the estimator itself unchanged:

* Rather than forming and inverting a 2J x 2J matrix, log|Sigma| and the
  quadratic form are evaluated in closed form from the eigenstructure of
  the covariance (see `rho_bounds` and `_neg_profile_nll`).
* Rather than running a general optimizer over both parameters, sigma2 is
  concentrated out analytically -- it has a closed-form conditional
  maximizer for any rho -- which reduces the fit to an exact
  one-dimensional search. This gives the same maximizer as a joint
  optimization but cannot stall or converge to a boundary. The paper's
  method-of-moments estimates are still computed (they are reported on
  the result as `sigma2_mom` / `rho_mom`, and seed the search), but the
  method-of-moments Wald test itself is not exposed: in our simulations
  it is badly anti-conservative at small rho (FPR up to 0.40 at rho=0.01,
  J=300, where its rho estimate is noisier than rho itself and the
  implied variance collapses).

This replaces an earlier closed-form REML-type estimator in this module
that was derived without access to the paper's estimating equations and
was anti-conservative (FPR up to ~3-4x nominal) for rho in [0.05, 0.2].
That failure mode is gone; see the numbers below.

How well it is calibrated (our own simulations, not the paper's)
----------------------------------------------------------------
All numbers below come from sampling D directly from the assumed
covariance structure with mu = 0 and sigma2 = 1, i.e. the paper's own toy
scheme, and are reproducible from `pydra_ml/tests/test_sharp.py`.

Replicating the paper's toy setup (J = 300, rho swept over the 49 values
0.01 ... 0.49, 100 draws each, alpha = 0.05) gives false positive rates
with median 0.040, minimum 0.000 and maximum 0.100 -- the tight cluster
around nominal that the paper reports for its score test, and nothing
like the 0.3-0.55 excursions it reports for the ML/ReML variants.

Because J is a free parameter here (`n_repeats`), calibration was also
measured on a grid of J against rho, 4000 draws per cell. Observed false
positive rate at alpha = 0.05:

    rho      J=10    J=20    J=30    J=60   J=100   J=200   J=300
    0.00    0.000   0.000   0.000   0.000   0.000   0.000   0.000
    0.01    0.000   0.000   0.000   0.000   0.000   0.000   0.000
    0.02    0.000   0.000   0.000   0.000   0.001   0.001   0.001
    0.03    0.000   0.000   0.000   0.000   0.001   0.003   0.004
    0.05    0.000   0.001   0.001   0.005   0.005   0.008   0.017
    0.07    0.000   0.003   0.005   0.008   0.015   0.022   0.030
    0.10    0.001   0.006   0.008   0.017   0.019   0.033   0.037
    0.12    0.003   0.008   0.010   0.022   0.025   0.039   0.047
    0.15    0.007   0.011   0.018   0.025   0.034   0.041   0.047
    0.17    0.005   0.018   0.024   0.035   0.039   0.040   0.046
    0.20    0.009   0.019   0.026   0.037   0.042   0.049   0.047
    0.25    0.018   0.029   0.040   0.047   0.050   0.042   0.046
    0.30    0.018   0.032   0.036   0.041   0.050   0.050   0.050
    0.35    0.028   0.044   0.045   0.050   0.049   0.054   0.049
    0.40    0.033   0.047   0.045   0.049   0.051   0.049   0.043
    0.45    0.046   0.048   0.044   0.051   0.050   0.044   0.049
    0.49    0.044   0.045   0.048   0.051   0.044   0.055   0.055

Reading that table:

* The test never over-rejects anywhere on the grid. The largest cell is
  0.055 and the largest upper end of a Wilson 95% interval is 0.062, both
  consistent with the nominal 0.05 at 4000 draws. Whatever else is true,
  this test does not manufacture false positives -- which is exactly the
  property the previous estimator lacked.
* It is *conservative*, sometimes drastically so, when the true
  between-repetition correlation is small, and the smaller J is the wider
  that region gets. The false positive rate is within about 20% of
  nominal for rho >= 0.35 at J = 10, rho >= 0.25 at J = 20-30,
  rho >= 0.15 at J = 60-100, and rho >= 0.12 at J = 200-300; below that
  it falls away, reaching effectively zero for rho <= 0.03 at every J
  tested.
* This is a property of the score test, not of a numerical shortcut:
  fixing mu at the null lets a large |D-bar| be explained by a larger
  rho, which inflates the standard error in step with the numerator.
  Conservative rejection is the safe direction for a significance claim,
  but it is bought with power.

The power cost is real. Setting mu to the value at which a test with
oracle knowledge of Var(D-bar) would have exactly 80% power, and
measuring what this test actually achieves (2000 draws per cell):

    rho      J=10    J=20    J=30    J=60   J=100   J=200
    0.00    0.011   0.003   0.003   0.000   0.000   0.000
    0.05    0.080   0.139   0.161   0.267   0.346   0.464
    0.10    0.202   0.340   0.388   0.538   0.570   0.680
    0.20    0.409   0.539   0.618   0.690   0.733      --
    0.30    0.550   0.662   0.721   0.735   0.779      --
    0.45    0.662   0.723   0.760   0.764   0.807      --

So a difference that an oracle test would detect 4 times in 5 may be
missed here nearly always when rho is near zero. Confidence-interval
coverage tracks the same pattern in the safe direction: over the same
grid it ranged from 0.942 to 1.000 against a nominal 0.95, i.e. never
materially narrower than advertised and often much wider.

One further property worth knowing: the statistic does not diverge. |Z|
plateaus at roughly sqrt(J + 1), so the two-sided p-value has a floor of
about 2 * (1 - Phi(sqrt(J + 1))) -- see `smallest_meaningful_p_value`.
At J = 2 that floor is above 0.05 and the test cannot reject at all, which
`sharp_test` warns about.

Correctness checks on the implementation itself, beyond the calibration
simulations: the closed-form log-determinant and quadratic form agree
with literal 2J x 2J `slogdet`/`solve` to floating-point precision, and
the one-dimensional fit agrees with an exhaustive 200,000-point scan of
the profile likelihood to within 1e-9 in log-likelihood (5e-15 in a wider
offline sweep over J = 3 ... 300). Both are asserted in
`pydra_ml/tests/test_sharp.py`.

Why the experimental gate stays
-------------------------------
`sharp_test` / `sharp_compare` still require `experimental=True`. The
reason has changed but has not gone away:

* This remains an independent reimplementation of a method from a preprint
  that has not been peer reviewed, with no reference implementation to
  check against. Agreement with the paper's published Fig. S11 pattern is
  evidence, not proof.
* More concretely, the test is simply not calibrated at its nominal level
  over much of the parameter space. A p-value from it is not a p-value in
  the usual sense unless the fitted rho is comfortably inside the
  near-nominal region above; elsewhere it is an upper bound on one, and
  the corresponding loss of power is severe.

What that means in practice: a *rejection* from this test is meaningful
(the test does not over-reject anywhere we measured), while a
*non-rejection* is weak evidence -- especially at small J or small fitted
rho, where the test has very little power. Check `result.rho` and
`result.n_repeats` against the table above before reading anything into a
null result.
"""

import dataclasses
import typing as ty
import warnings

import numpy as np
from scipy import optimize, stats


@dataclasses.dataclass
class SharpTestResult:
    """Result of the SHARP test for a mean cross-validated performance
    difference between two models.

    :ivar mean_diff: D-bar, the estimated mean performance difference
        (model 1 minus model 2). This is both the GLS and the ML estimate
        of mu, and equals the plain average of all 2J half-statistics.
    :ivar z: test statistic, asymptotically standard normal under the null.
    :ivar p_value: two-sided p-value, 2 * (1 - Phi(|z|)).
    :ivar ci: (1 - alpha) confidence interval, obtained by inverting the
        same test (the set of mu0 that would not be rejected).
    :ivar sigma2: fitted per-repetition variance under the null.
    :ivar rho: fitted between-repetition correlation under the null.
    :ivar n_repeats: J, the number of split-half repetitions.
    :ivar sigma2_mom: method-of-moments variance estimate (always reported;
        it is what the score test's optimizer is initialized from).
    :ivar rho_mom: method-of-moments correlation estimate, unclipped, so it
        can fall outside the range in which Sigma is positive definite.
    """

    mean_diff: float
    z: float
    p_value: float
    ci: ty.Tuple[float, float]
    sigma2: float
    rho: float
    n_repeats: int
    sigma2_mom: float
    rho_mom: float


def rho_bounds(n_repeats):
    """Open interval of rho over which the SHARP covariance matrix
    Sigma(sigma2, rho) is positive definite.

    Sigma = sigma2 * [[M, C], [C, M]] is block-symmetric, so the orthogonal
    sum/difference rotation block-diagonalizes it into M + C and M - C.
    Here M - C = I (the rho terms cancel exactly), and M + C =
    (1 - 2*rho) * I + 2*rho * 11^T has eigenvalues 1 + 2*(J - 1)*rho (on the
    all-ones direction) and 1 - 2*rho (with multiplicity J - 1). Positive
    definiteness therefore requires -1 / (2 * (J - 1)) < rho < 1/2. Note the
    upper limit is 1/2, not 1: because the same rho couples the A and B
    halves as well as repetitions within a half, correlations of 1/2 or more
    are not representable by this model. (This is consistent with the
    paper's own toy simulation, which sweeps rho only up to 0.49.)
    """
    return -1.0 / (2.0 * (n_repeats - 1)), 0.5


def _sufficient_stats(diff_a, diff_b):
    """Reduce the 2J half-statistics to the quantities the likelihood
    depends on.

    With S_j = D_Aj + D_Bj and Delta_j = D_Aj - D_Bj, and using the
    rotation described in `rho_bounds`, the Gaussian log-likelihood depends
    on the data only through D-bar, sum_j (S_j - S-bar)^2 and
    sum_j Delta_j^2 (plus, for the method of moments, the two within-half
    sample variances).

    :return: (d_bar, ss_s, ss_delta, s_a2, s_b2)
    """
    s = diff_a + diff_b
    delta = diff_a - diff_b
    d_bar = s.mean() / 2.0
    ss_s = float(np.sum((s - s.mean()) ** 2))
    ss_delta = float(np.sum(delta**2))
    s_a2 = float(np.var(diff_a, ddof=1))
    s_b2 = float(np.var(diff_b, ddof=1))
    return float(d_bar), ss_s, ss_delta, s_a2, s_b2


def _mom_estimates(n_repeats, ss_delta, s_a2, s_b2):
    """Method-of-moments estimates of (sigma2, rho).

    E[sigma_delta^2] = sigma2 with sigma_delta^2 = (1 / (2J)) *
    sum_j (D_Aj - D_Bj)^2 (no centering: E[D_Aj - D_Bj] = 0 by
    construction), and E[S_A^2] = E[S_B^2] = sigma2 * (1 - rho) for the
    within-half sample variances, which inverts to the estimates below.
    rho is returned unclipped and can therefore be outside the range from
    `rho_bounds`.
    """
    sigma2 = ss_delta / (2.0 * n_repeats)
    if sigma2 <= 0.0:
        return 0.0, 0.0
    rho = (sigma2 - 0.5 * (s_a2 + s_b2)) / sigma2
    return sigma2, rho


def var_d_bar(n_repeats, sigma2, rho):
    """Var(D-bar) = sigma2 * (1 / (2J) + (J - 1) / J * rho)."""
    return sigma2 * (1.0 / (2 * n_repeats) + (n_repeats - 1) / n_repeats * rho)


def smallest_meaningful_p_value(n_repeats):
    """Floor on the score test's two-sided p-value at J repetitions.

    The score statistic does not diverge as the observed difference grows:
    re-fitting the nuisance parameters with the mean pinned at the null
    lets a large |D-bar| be explained by a larger rho, which inflates the
    standard error in step with the numerator. Working through the large-
    |D-bar| limit of the profile likelihood, the stationary point sits at
    b = 1 - 2*rho proportional to 1 / (D-bar)^2, at which z^2 -> J + 1.

    So |z| plateaus at about sqrt(J + 1) and the two-sided p-value cannot
    meaningfully go below 2 * (1 - Phi(sqrt(J + 1))) -- roughly 9e-4 at
    J = 10 and 3e-8 at J = 30. Reported p-values below this floor come from
    the numerical clamp that keeps rho inside its positive-definite range
    rather than from the model, and should be read as "at the floor", not
    as a quantitative significance level. At J = 2 the floor (0.083) is
    above the usual alpha = 0.05, i.e. the test cannot reject at all.
    """
    return float(2 * stats.norm.sf(np.sqrt(n_repeats + 1.0)))


def _neg_profile_nll(rho, n_repeats, t, ss_s, ss_delta):
    """Negative profile log-likelihood in rho, with sigma2 concentrated out.

    Under the rotation described in `rho_bounds`, with a = 1 + 2*(J-1)*rho
    and b = 1 - 2*rho, and t = D-bar - mu (the mean held fixed at the value
    being tested),

        log|Sigma| = 2J*log(sigma2) + log(a) + (J-1)*log(b)
        (D - mu*1)^T Sigma^-1 (D - mu*1) = Q(rho) / sigma2, where
        Q(rho) = 2J*t^2 / a + (1/2)*ss_s / b + (1/2)*ss_delta

    For any fixed rho the likelihood is maximized at sigma2 = Q(rho) / (2J),
    which leaves the profile objective returned here (up to an additive
    constant). Because concentrating sigma2 out is exact, maximizing this
    over rho gives exactly the same (sigma2, rho) as jointly optimizing both
    parameters, but as a well-behaved one-dimensional problem.

    `rho` may be a scalar or an array (the array form is used to scan the
    grid that brackets the optimum).
    """
    rho = np.asarray(rho, dtype=float)
    a = 1.0 + 2.0 * rho * (n_repeats - 1)
    b = 1.0 - 2.0 * rho
    q = 2.0 * n_repeats * t * t / a + 0.5 * ss_s / b + 0.5 * ss_delta
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (
            n_repeats * np.log(q / (2.0 * n_repeats))
            + 0.5 * np.log(a)
            + 0.5 * (n_repeats - 1) * np.log(b)
        )
    out = np.where(q > 0.0, out, np.inf)
    return out if out.ndim else float(out)


def _restricted_mle(n_repeats, t, ss_s, ss_delta, rho_init):
    """Maximize the Gaussian log-likelihood over (sigma2, rho) with the mean
    held fixed (mu = mu0, so t = D-bar - mu0 is fixed too).

    A coarse grid over the positive-definite range of rho brackets the
    optimum -- guarding against the optimizer walking into a boundary from a
    poor starting point -- and a bounded Brent search refines it. The
    method-of-moments value is included as a grid candidate so that the
    search is initialized from it in the same spirit as the paper's
    implementation.

    :return: (sigma2_hat, rho_hat)
    """
    lo, hi = rho_bounds(n_repeats)
    margin = 1e-6 * (hi - lo)
    lo, hi = lo + margin, hi - margin

    grid = np.linspace(lo, hi, 129)
    if np.isfinite(rho_init) and lo < rho_init < hi:
        grid = np.sort(np.append(grid, rho_init))
    values = _neg_profile_nll(grid, n_repeats, t, ss_s, ss_delta)
    k = int(np.argmin(values))
    left = grid[max(k - 1, 0)]
    right = grid[min(k + 1, grid.size - 1)]
    if right > left:
        res = optimize.minimize_scalar(
            _neg_profile_nll,
            bounds=(left, right),
            args=(n_repeats, t, ss_s, ss_delta),
            method="bounded",
            options={"xatol": 1e-10},
        )
        rho_hat = float(res.x) if res.fun <= values[k] else float(grid[k])
    else:
        rho_hat = float(grid[k])

    a = 1.0 + 2.0 * rho_hat * (n_repeats - 1)
    b = 1.0 - 2.0 * rho_hat
    q = 2.0 * n_repeats * t * t / a + 0.5 * ss_s / b + 0.5 * ss_delta
    sigma2_hat = q / (2.0 * n_repeats)
    return sigma2_hat, rho_hat


def _score_z(n_repeats, t, ss_s, ss_delta, rho_init):
    """Score statistic for H0: mu = D-bar - t, with the nuisance parameters
    re-estimated under that null.

    :return: (z, sigma2_hat, rho_hat)
    """
    sigma2_hat, rho_hat = _restricted_mle(n_repeats, t, ss_s, ss_delta, rho_init)
    variance = var_d_bar(n_repeats, sigma2_hat, rho_hat)
    if not variance > 0.0:
        return (0.0 if t == 0.0 else np.copysign(np.inf, t)), sigma2_hat, rho_hat
    return t / np.sqrt(variance), sigma2_hat, rho_hat


def _invert_test(z_of_t, z_crit, t_scale):
    """Half-width of the confidence interval: the smallest t > 0 at which the
    statistic reaches `z_crit`.

    The statistic depends on the hypothesized mean only through
    t = D-bar - mu0 (the other sufficient statistics are mean-free), and is
    odd in t, so the interval is symmetric about D-bar and only one root has
    to be found. `np.inf` is returned when the statistic never reaches
    z_crit -- see the module docstring on the score statistic's finite
    supremum at small J.
    """
    hi = max(t_scale, 1e-12)
    for _ in range(200):
        if z_of_t(hi) >= z_crit:
            break
        hi *= 2.0
    else:
        return np.inf
    return float(
        optimize.brentq(
            lambda t: z_of_t(t) - z_crit, 0.0, hi, xtol=max(1e-14, 1e-10 * hi)
        )
    )


def _require_experimental(experimental):
    if not experimental:
        raise ValueError(
            "The SHARP test is experimental: it is an independent "
            "reimplementation of a preprint's method, validated only by our "
            "own simulations, and it is markedly conservative (so also "
            "under-powered) at small between-repetition correlation. See "
            "pydra_ml/sharp_test.py's module docstring for the calibration "
            "table. Call with experimental=True to acknowledge this and "
            "proceed."
        )


def sharp_test(diff_a, diff_b, alpha=0.05, experimental=False):
    """SHARP test for a mean cross-validated performance difference.

    EXPERIMENTAL -- must be called with experimental=True. The module
    docstring has the calibration table this was validated against: the
    test does not over-reject anywhere we measured, but it is markedly
    conservative (and correspondingly under-powered) when the fitted `rho`
    is small, so read `result.rho` and `result.n_repeats` against that
    table before drawing anything from a non-rejection.

    :param diff_a: Per-repetition fold-averaged performance differences
        (model 1 minus model 2) from split half A. Length J.
    :param diff_b: Same, from split half B. Length J.
    :param alpha: Significance level for the confidence interval.
    :param experimental: Must be set to True to acknowledge this is an
        experimental implementation (see the module docstring).
    :return: SharpTestResult with the mean difference, Z-score, two-sided
        p-value, a (1 - alpha) confidence interval, and the fitted variance
        (sigma2) and between-repetition correlation (rho).
    """
    _require_experimental(experimental)
    diff_a = np.asarray(diff_a, dtype=float)
    diff_b = np.asarray(diff_b, dtype=float)
    if diff_a.shape != diff_b.shape or diff_a.ndim != 1:
        raise ValueError("diff_a and diff_b must be 1-D arrays of the same length")
    n_repeats = diff_a.size
    if n_repeats < 2:
        raise ValueError("SHARP requires at least 2 repetitions (J >= 2)")
    p_floor = smallest_meaningful_p_value(n_repeats)
    if p_floor >= alpha:
        warnings.warn(
            f"With J={n_repeats} repetitions the SHARP score statistic "
            f"plateaus below the alpha={alpha} rejection threshold (the "
            f"smallest meaningful p-value is {p_floor:.3g}), so this test "
            f"cannot reject the null however large the observed difference "
            f"is. Use more repetitions.",
            stacklevel=2,
        )

    d_bar, ss_s, ss_delta, s_a2, s_b2 = _sufficient_stats(diff_a, diff_b)
    sigma2_mom, rho_mom = _mom_estimates(n_repeats, ss_delta, s_a2, s_b2)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    def z_of_t(t):
        return _score_z(n_repeats, t, ss_s, ss_delta, rho_mom)[0]

    z, sigma2_hat, rho_hat = _score_z(n_repeats, d_bar, ss_s, ss_delta, rho_mom)
    variance = max(var_d_bar(n_repeats, sigma2_hat, rho_hat), 0.0)
    half_width = _invert_test(z_of_t, z_crit, z_crit * np.sqrt(variance))

    p_value = 2 * stats.norm.sf(np.abs(z))
    return SharpTestResult(
        mean_diff=float(d_bar),
        z=float(z),
        p_value=float(p_value),
        ci=(float(d_bar - half_width), float(d_bar + half_width)),
        sigma2=float(sigma2_hat),
        rho=float(rho_hat),
        n_repeats=int(n_repeats),
        sigma2_mom=float(sigma2_mom),
        rho_mom=float(rho_mom),
    )


def _score_predictions(metric, y_true, y_pred, y_proba):
    """Compute a named sklearn.metrics score, mirroring tasks.calc_metric's
    convention: roc_auc_score uses predicted probabilities of the positive
    class when available, everything else uses hard predictions."""
    import sklearn.metrics

    metric_func = getattr(sklearn.metrics, metric)
    if metric == "roc_auc_score" and y_proba is not None:
        return metric_func(y_true, y_proba[:, 1])
    return metric_func(y_true, y_pred)


def split_half_repeated_cv(
    X,
    y,
    clf_info_1,
    clf_info_2,
    metric,
    n_repeats=30,
    n_folds=5,
    random_state=None,
):
    """Generate the paired half-statistics SHARP needs, by repeatedly
    splitting (X, y) into two disjoint halves and running K-fold
    cross-validation independently within each half.

    In each of `n_repeats` repetitions: split the samples into two disjoint
    halves; within each half, run `n_folds`-fold cross-validation for both
    clf_info_1 and clf_info_2, scoring each fold with `metric`; average the
    per-fold (model 1 - model 2) score difference over the folds in that
    half.

    :param X: Input features
    :param y: Target variable
    :param clf_info_1: clf_info for the first model (see the clf_info spec
        format in the README)
    :param clf_info_2: clf_info for the second model
    :param metric: name of a function in sklearn.metrics (e.g.
        "roc_auc_score", "accuracy_score") -- the same convention used
        elsewhere in this package's spec files (not an sklearn scorer
        registry key).
    :param n_repeats: number of split-half repetitions (J in sharp_test)
    :param n_folds: number of cross-validation folds within each half
    :param random_state: seed for the half splits and fold splits
    :return: (diff_a, diff_b), each a length-`n_repeats` array, suitable for
        sharp_test
    """
    from sklearn.model_selection import KFold

    from .tasks import build_pipeline

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)

    def fit_and_score(clf_info, train_idx, test_idx):
        pipe = build_pipeline(clf_info)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        try:
            y_proba = pipe.predict_proba(X[test_idx])
        except AttributeError:
            y_proba = None
        return _score_predictions(metric, y[test_idx], y_pred, y_proba)

    def half_fold_diff(indices):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(2**32 - 1))
        fold_diffs = []
        for train_idx, test_idx in kf.split(indices):
            train_idx, test_idx = indices[train_idx], indices[test_idx]
            score_1 = fit_and_score(clf_info_1, train_idx, test_idx)
            score_2 = fit_and_score(clf_info_2, train_idx, test_idx)
            fold_diffs.append(score_1 - score_2)
        return np.mean(fold_diffs)

    diff_a = np.empty(n_repeats)
    diff_b = np.empty(n_repeats)
    for j in range(n_repeats):
        perm = rng.permutation(n_samples)
        half = n_samples // 2
        indices_a, indices_b = perm[:half], perm[half:]
        diff_a[j] = half_fold_diff(indices_a)
        diff_b[j] = half_fold_diff(indices_b)
    return diff_a, diff_b


def sharp_compare(
    X,
    y,
    clf_info_1,
    clf_info_2,
    metric,
    n_repeats=30,
    n_folds=5,
    alpha=0.05,
    random_state=None,
    experimental=False,
):
    """Compare two models with the SHARP test, running the split-half
    repeated cross-validation procedure and the test in one call.

    EXPERIMENTAL -- see the module docstring and sharp_test for the
    calibration this was validated against. Must be called with
    experimental=True.

    See `split_half_repeated_cv` and `sharp_test` for parameter details.

    :return: SharpTestResult
    """
    # Check the gate before the cross-validation rather than after it: the
    # model fits below are the expensive part, and there is no point paying
    # for them only to refuse the test.
    _require_experimental(experimental)
    diff_a, diff_b = split_half_repeated_cv(
        X,
        y,
        clf_info_1,
        clf_info_2,
        metric,
        n_repeats=n_repeats,
        n_folds=n_folds,
        random_state=random_state,
    )
    return sharp_test(diff_a, diff_b, alpha=alpha, experimental=experimental)
