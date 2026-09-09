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
EXPERIMENTAL -- DO NOT USE FOR A PUBLICATION-QUALITY SIGNIFICANCE CLAIM
======================================================================

The paper's Methods Section 4.6 gives the model (the covariance structure
summarized above), the SHARP estimator and its variance formula, and
states that all reported results use a "score test" whose Z-score is D̄
over a standard error using null-restricted (mu=0) maximum-likelihood
estimates of sigma2 and rho. It does not give the exact estimating
equations (those are in Supplementary Methods S7, which was not
reachable while implementing this: biorxiv.org blocks automated access to
the article itself and, separately, to its supplementary-material
download, and no supplementary content was exposed via PMC or the
bioRxiv API). This module's covariance structure and estimators are
therefore our own derivation from the stated model, not a port of the
paper's own code.

An independent statistician review confirmed the covariance-structure
derivation, but found that naive maximum-likelihood estimation of
(sigma2, rho) is structurally flawed here in two ways: (1) null-restricted
joint ML of (mu=0, sigma2, rho) makes the test statistic self-referential
(its variance estimate is driven by the same residual being tested); (2)
unrestricted/REML ML frequently converges to (or past) the boundary where
Var(D̄) is exactly zero, which can silently collapse the confidence
interval to near-zero width. This module uses a closed-form REML-type
estimator instead (no numerical optimizer, so no convergence risk;
doesn't depend on the residual being tested, so it isn't
self-referential), floored at rho=0 (correlation from repeated reuse of
overlapping data should not be negative) to keep Var(D̄) away from its
zero boundary.

That floor avoids the worst failure (the collapsed CI), but leaves a
real, uncorrected miscalibration behind: simulating directly from the
assumed covariance structure (pydra_ml/tests/test_sharp.py) shows the
false positive rate is near the nominal alpha when the true
between-repetition correlation is 0 or larger (roughly rho <= 0 or
rho >= 0.3 in our simulations), but *anti-conservative* -- up to ~3-4x
the nominal alpha -- for small positive correlation (roughly rho in
[0.05, 0.2]), which is arguably the most common real-world case
(repeated CV on overlapping data usually induces at least mild positive
correlation). We tried three different bootstrap-calibration schemes to
correct this and none reliably did (one made rho=0 much worse); properly
closing this gap looks like the kind of boundary-corrected reference
distribution problem covered in, e.g., Self & Liang (1987) for testing a
variance component at its boundary, which is beyond what we could derive
and validate here.

Given this, `sharp_test`/`sharp_compare` require `experimental=True` to
call, so this can't be reached accidentally or hold up as a validated
default. Treat any result as a rough, unverified signal alongside other
evidence, not a standalone significance claim -- and if you can get the
paper's Supplementary Methods S7 (e.g. by asking the corresponding
author), we would very much like to fix this properly.
"""

import dataclasses
import typing as ty

import numpy as np
from scipy import stats


@dataclasses.dataclass
class SharpTestResult:
    """Result of the SHARP test for a mean cross-validated performance
    difference between two models."""

    mean_diff: float
    z: float
    p_value: float
    ci: ty.Tuple[float, float]
    sigma2: float
    rho: float
    n_repeats: int


def _fit_sigma2_rho(J, ss_s, ss_delta):
    """Closed-form REML-type estimate of (sigma2, rho), floored at rho=0.

    `ss_s` and `ss_delta` are the sum-of-squares of, respectively, S_j =
    diff_a[j] + diff_b[j] (deviations from its own mean) and Delta_j =
    diff_a[j] - diff_b[j] (already mean zero by construction). Neither
    depends on the mean performance difference mu, so this estimate does
    not depend on the value being tested (avoiding the self-referential
    degeneracy of a null-restricted or profiled joint ML fit -- see the
    module docstring).

    Using the per-direction unbiased estimates sigma2_hat = ss_delta / (2J)
    (J degrees of freedom) and lambda2_hat = ss_s / (J - 1) (J - 1 degrees
    of freedom), the implied unconstrained estimate of lambda1 = Var(S_j)
    + (J - 1) * Cov(S_j, S_j') reduces to the exact closed form
    lambda1_hat = ss_delta - ss_s (derived from lambda1 = 2*J*sigma2 -
    (J - 1)*lambda2). When this is smaller than lambda1 at rho=0 --
    which happens often when the true rho is small, since ss_s and
    ss_delta are then just noisy estimates of the same quantity 2*sigma2
    -- the closed-form optimum lies at rho < 0, which is floored to 0
    (Var(D̄) is otherwise driven towards its zero boundary; see the module
    docstring).

    :return: (sigma2_hat, rho_hat, var_d_bar), where var_d_bar is
        Var(D̄) = sigma2_hat * (1 / (2J) + (J - 1) / J * rho_hat) evaluated
        at these estimates.
    """
    sigma2_unc = ss_delta / (2 * J)
    lam1_unc = ss_delta - ss_s
    lam1_floor = 2 * sigma2_unc  # lambda1 at rho=0
    if lam1_unc >= lam1_floor:
        sigma2_hat = sigma2_unc
        rho_hat = (lam1_unc / (2 * sigma2_unc) - 1) / (2 * (J - 1))
        var_d_bar = lam1_unc / (4 * J)
    else:
        rho_hat = 0.0
        sigma2_hat = (ss_s + ss_delta) / (2 * (2 * J - 1))
        var_d_bar = sigma2_hat / (2 * J)
    return sigma2_hat, rho_hat, var_d_bar


def sharp_test(diff_a, diff_b, alpha=0.05, experimental=False):
    """SHARP test for a mean cross-validated performance difference.

    EXPERIMENTAL -- see the module docstring for a known, uncorrected
    anti-conservative miscalibration region (small positive between-
    repetition correlation). Must be called with experimental=True.

    :param diff_a: Per-repetition fold-averaged performance differences
        (model 1 minus model 2) from split half A. Length J.
    :param diff_b: Same, from split half B. Length J.
    :param alpha: Significance level for the confidence interval.
    :param experimental: Must be set to True to acknowledge this is an
        experimental, not-fully-validated implementation (see the module
        docstring).
    :return: SharpTestResult with the mean difference, Z-score, two-sided
        p-value, a (1 - alpha) confidence interval, and the fitted
        variance (sigma2) and between-repetition correlation (rho).
    """
    if not experimental:
        raise ValueError(
            "sharp_test is experimental and not fully validated (see the "
            "module docstring for a known anti-conservative miscalibration "
            "region). Call with experimental=True to acknowledge this and "
            "proceed."
        )
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

    sigma2_hat, rho_hat, var_d_bar = _fit_sigma2_rho(J, ss_s, ss_delta)
    z = d_bar / np.sqrt(var_d_bar)
    p_value = 2 * stats.norm.sf(np.abs(z))
    margin = stats.norm.ppf(1 - alpha / 2) * np.sqrt(var_d_bar)

    return SharpTestResult(
        mean_diff=float(d_bar),
        z=float(z),
        p_value=float(p_value),
        ci=(float(d_bar - margin), float(d_bar + margin)),
        sigma2=float(sigma2_hat),
        rho=float(rho_hat),
        n_repeats=J,
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

    EXPERIMENTAL -- see the module docstring and sharp_test for a known,
    uncorrected anti-conservative miscalibration region. Must be called
    with experimental=True.

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
    return sharp_test(diff_a, diff_b, alpha=alpha, experimental=experimental)
