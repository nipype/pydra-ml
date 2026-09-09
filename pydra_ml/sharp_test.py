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

CAVEAT: the paper's Methods Section 4.6 gives the model (the covariance
structure summarized above), the SHARP estimator and its variance formula,
and states that all reported results use a "score test" whose Z-score is
D̄ over a standard error using null-restricted (mu=0) maximum-likelihood
estimates of sigma2 and rho. It does not give the exact estimating
equations (those are in Supplementary Methods S7, which was not
reachable while implementing this: biorxiv.org blocks automated access,
and no supplementary-file link was exposed via PMC or the bioRxiv API).
This module's covariance structure, likelihood, and estimators are
therefore our own derivation from the stated model, not a port of the
paper's own code.

We validated this implementation by simulating directly from the assumed
covariance structure (see pydra_ml/tests/test_sharp.py) and checking false
positive rates under the null. It is well-calibrated (false positive rate
near the nominal alpha) when the fitted between-repetition correlation
(rho) is moderate to large (roughly >= 0.3 in our simulations), and
conservative -- reduced false positive rate *and* reduced power, never
anti-conservative -- when rho is small (near its structural lower bound
of -1 / (2 * (J - 1))). We were not able to identify a fix that resolves
this without access to the paper's own derivation; a method-of-moments
alternative we tried was worse (frequently negative variance estimates
requiring clipping, which made the test anti-conservative instead). Until
this is resolved, prefer this test when you expect non-trivial
correlation between repetitions (the common case when repetitions reuse
overlapping data, which is the whole reason SHARP exists), and treat a
result with rho close to its lower bound with extra caution.
"""

import dataclasses
import typing as ty

import numpy as np
from scipy import optimize, stats


@dataclasses.dataclass
class SharpTestResult:
    """Result of the SHARP test for a mean cross-validated performance
    difference between two models."""

    mean_diff: float
    z: float
    p_value: float
    ci: ty.Tuple[float, float]
    sigma2_null: float
    rho_null: float
    sigma2: float
    rho: float
    n_repeats: int


def _log_lik_null(sigma2, rho, J, s_bar, ss_s, ss_delta):
    """Gaussian log-likelihood (up to an additive constant) for the SHARP
    covariance structure under the null hypothesis mu=0, as a function of
    (sigma2, rho).

    `s_bar`, `ss_s` and `ss_delta` are sufficient statistics of, respectively,
    the mean, the within-sum-of-squares of S_j = diff_a[j] + diff_b[j], and
    the sum-of-squares of Delta_j = diff_a[j] - diff_b[j] (already mean zero
    by construction, regardless of the true mu). With mu fixed at 0, s_bar
    is a genuine residual from its null-hypothesis mean of 0.
    """
    lam1 = 2 * sigma2 * (1 + 2 * rho * (J - 1))
    lam2 = 2 * sigma2 * (1 - 2 * rho)
    if lam1 <= 0 or lam2 <= 0 or sigma2 <= 0:
        return -np.inf
    ll = -0.5 * np.log(lam1 / J) - 0.5 * J * s_bar**2 / lam1
    ll += -0.5 * (J - 1) * np.log(lam2) - 0.5 * ss_s / lam2
    ll += -0.5 * J * np.log(2 * sigma2) - 0.5 * ss_delta / (2 * sigma2)
    return ll


def _log_lik_reml(sigma2, rho, J, ss_s, ss_delta):
    """Restricted (residual) log-likelihood for (sigma2, rho), leaving mu
    unrestricted.

    Ordinary (non-restricted) ML estimation of (sigma2, rho) with mu also
    free is degenerate here: profiling mu out exactly zero-fits the
    mean-direction residual for any (sigma2, rho), so the profiled
    likelihood has no interior maximum and diverges towards the boundary
    rho -> -1 / (2 * (J - 1)) (where the variance of the mean estimator
    goes to 0). REML avoids this by dropping the mean-direction term
    entirely and fitting (sigma2, rho) only from the directions of the
    data orthogonal to the mean (ss_s, ss_delta), which don't depend on mu.
    """
    lam2 = 2 * sigma2 * (1 - 2 * rho)
    if lam2 <= 0 or sigma2 <= 0:
        return -np.inf
    ll = -0.5 * (J - 1) * np.log(lam2) - 0.5 * ss_s / lam2
    ll += -0.5 * J * np.log(2 * sigma2) - 0.5 * ss_delta / (2 * sigma2)
    return ll


def _fit_rho_bounds(J):
    """Range of rho for which both eigenvalues of the SHARP covariance
    structure (lam1, lam2) are positive."""
    return -1.0 / (2 * (J - 1)), 0.5


def _minimize_sigma2_rho(neg_ll, J, sigma2_0):
    """Minimize `neg_ll(sigma2, rho)`, reparameterized to an unconstrained
    optimization so rho stays strictly inside its valid range."""
    rho_lo, rho_hi = _fit_rho_bounds(J)

    def unpack(params):
        log_sigma2, rho_logit = params
        sigma2 = np.exp(log_sigma2)
        rho = rho_lo + (rho_hi - rho_lo) / (1 + np.exp(-rho_logit))
        return sigma2, rho

    def neg_ll_reparam(params):
        return neg_ll(*unpack(params))

    result = optimize.minimize(
        neg_ll_reparam, x0=[np.log(sigma2_0), 0.0], method="Nelder-Mead"
    )
    return unpack(result.x)


def _fit_sigma2_rho_null(J, s_bar, ss_s, ss_delta):
    """Null-restricted (mu=0) MLE of (sigma2, rho), used for the score test."""
    sigma2_0 = max(ss_delta / (2 * J), 1e-8)
    return _minimize_sigma2_rho(
        lambda sigma2, rho: -_log_lik_null(sigma2, rho, J, s_bar, ss_s, ss_delta),
        J,
        sigma2_0,
    )


def _fit_sigma2_rho_reml(J, ss_s, ss_delta):
    """REML estimate of (sigma2, rho) with mu unrestricted, used for the CI."""
    sigma2_0 = max(ss_delta / (2 * J), 1e-8)
    return _minimize_sigma2_rho(
        lambda sigma2, rho: -_log_lik_reml(sigma2, rho, J, ss_s, ss_delta),
        J,
        sigma2_0,
    )


def sharp_test(diff_a, diff_b, alpha=0.05):
    """SHARP score test for a mean cross-validated performance difference.

    :param diff_a: Per-repetition fold-averaged performance differences
        (model 1 minus model 2) from split half A. Length J.
    :param diff_b: Same, from split half B. Length J.
    :param alpha: Significance level for the confidence interval.
    :return: SharpTestResult with the mean difference, Z-score, two-sided
        p-value, a (1 - alpha) confidence interval, and the fitted variance
        (sigma2) and between-repetition correlation (rho), both under the
        null hypothesis (mean_diff=0, used for the p-value) and unrestricted
        (used for the confidence interval).
    """
    diff_a = np.asarray(diff_a, dtype=float)
    diff_b = np.asarray(diff_b, dtype=float)
    if diff_a.shape != diff_b.shape or diff_a.ndim != 1:
        raise ValueError("diff_a and diff_b must be 1-D arrays of the same length")
    J = diff_a.size
    if J < 2:
        raise ValueError("SHARP requires at least 2 repetitions (J >= 2)")

    s = diff_a + diff_b
    delta = diff_a - diff_b
    s_bar = s.mean()
    ss_s = np.sum((s - s_bar) ** 2)
    ss_delta = np.sum(delta**2)
    d_bar = s_bar / 2  # SHARP (GLS) estimator of the mean performance difference

    sigma2_null, rho_null = _fit_sigma2_rho_null(J, s_bar, ss_s, ss_delta)
    var_null = sigma2_null * (1 / (2 * J) + (J - 1) / J * rho_null)
    z = d_bar / np.sqrt(var_null)
    p_value = 2 * stats.norm.sf(np.abs(z))

    sigma2, rho = _fit_sigma2_rho_reml(J, ss_s, ss_delta)
    var_unrestricted = sigma2 * (1 / (2 * J) + (J - 1) / J * rho)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    margin = z_crit * np.sqrt(var_unrestricted)

    return SharpTestResult(
        mean_diff=float(d_bar),
        z=float(z),
        p_value=float(p_value),
        ci=(float(d_bar - margin), float(d_bar + margin)),
        sigma2_null=float(sigma2_null),
        rho_null=float(rho_null),
        sigma2=float(sigma2),
        rho=float(rho),
        n_repeats=J,
    )


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
        "roc_auc_score", "accuracy_score")
    :param n_repeats: number of split-half repetitions (J in sharp_test)
    :param n_folds: number of cross-validation folds within each half
    :param random_state: seed for the half splits and fold splits
    :return: (diff_a, diff_b), each a length-`n_repeats` array, suitable for
        sharp_test
    """
    from sklearn.metrics import get_scorer
    from sklearn.model_selection import KFold

    from .tasks import build_pipeline

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples = X.shape[0]
    scorer = get_scorer(metric)
    rng = np.random.RandomState(random_state)

    def half_fold_diff(indices):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(2**32 - 1))
        fold_diffs = []
        for train_idx, test_idx in kf.split(indices):
            train_idx, test_idx = indices[train_idx], indices[test_idx]
            score_1 = _fit_and_score(clf_info_1, X, y, train_idx, test_idx, scorer)
            score_2 = _fit_and_score(clf_info_2, X, y, train_idx, test_idx, scorer)
            fold_diffs.append(score_1 - score_2)
        return np.mean(fold_diffs)

    def _fit_and_score(clf_info, X, y, train_idx, test_idx, scorer):
        pipe = build_pipeline(clf_info)
        pipe.fit(X[train_idx], y[train_idx])
        return scorer(pipe, X[test_idx], y[test_idx])

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
):
    """Compare two models with the SHARP test, running the split-half
    repeated cross-validation procedure and the test in one call.

    See `split_half_repeated_cv` and `sharp_test` for parameter details.

    :return: SharpTestResult
    """
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
    return sharp_test(diff_a, diff_b, alpha=alpha)
