import os

import numpy as np
import pytest
from scipy import stats

from ..sharp_test import (
    _mom_estimates,
    _neg_profile_nll,
    _restricted_mle,
    _score_z,
    _sufficient_stats,
    rho_bounds,
    sharp_compare,
    sharp_test,
    smallest_meaningful_p_value,
    var_d_bar,
)


def _sharp_covariance(J, sigma2, rho):
    """The covariance SHARP assumes for D = [D_A; D_B]: Cov(D_Aj, D_Bj) = 0
    (same repetition, disjoint halves), Cov(anything else) = rho * sigma2,
    Var(each entry) = sigma2."""
    n = 2 * J
    cov = np.full((n, n), rho * sigma2)
    np.fill_diagonal(cov, sigma2)
    for j in range(J):
        cov[j, J + j] = 0.0
        cov[J + j, j] = 0.0
    return cov


def _simulate_sharp_data(J, mu, sigma2, rho, rng, size=None):
    """Draw `size` (default 1) independent realizations from that covariance.

    Sampling goes through the same rotation the estimator uses -- the
    sum/difference basis, in which the covariance is diagonal with
    eigenvalues sigma2 * (1 + 2*(J-1)*rho) once, sigma2 * (1 - 2*rho) with
    multiplicity J - 1, and sigma2 with multiplicity J -- so it costs O(J)
    per draw instead of a 2J x 2J Cholesky. `test_simulator_matches_the_
    assumed_covariance` checks the two agree.
    """
    n = 1 if size is None else size
    a = 1.0 + 2.0 * rho * (J - 1)
    b = 1.0 - 2.0 * rho
    z = rng.standard_normal((n, J))
    zbar = z.mean(axis=1, keepdims=True)
    s = np.sqrt(sigma2) * (np.sqrt(b) * (z - zbar) + np.sqrt(a) * zbar)
    s += np.sqrt(2.0) * mu
    d = np.sqrt(sigma2) * rng.standard_normal((n, J))
    diff_a = (s + d) / np.sqrt(2.0)
    diff_b = (s - d) / np.sqrt(2.0)
    if size is None:
        return diff_a[0], diff_b[0]
    return diff_a, diff_b


def _p_values(J, mu, sigma2, rho, n_sims, rng):
    """p-values for n_sims independent draws.

    Goes through `_score_z` rather than `sharp_test` so the Monte Carlo
    loops skip the (much more expensive) confidence-interval inversion;
    `test_p_value_matches_the_public_api` pins the two together.
    """
    diff_a, diff_b = _simulate_sharp_data(J, mu, sigma2, rho, rng, size=n_sims)
    out = np.empty(n_sims)
    for i in range(n_sims):
        d_bar, ss_s, ss_delta, s_a2, s_b2 = _sufficient_stats(diff_a[i], diff_b[i])
        _, rho_mom = _mom_estimates(J, ss_delta, s_a2, s_b2)
        z = _score_z(J, d_bar, ss_s, ss_delta, rho_mom)[0]
        out[i] = 2 * stats.norm.sf(abs(z))
    return out


def test_sharp_test_requires_experimental_acknowledgement():
    with pytest.raises(ValueError, match="experimental"):
        sharp_test(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="experimental"):
        sharp_compare(None, None, None, None, "accuracy_score")


def test_sharp_test_input_validation():
    with pytest.raises(ValueError):
        sharp_test(np.zeros(5), np.zeros(4), experimental=True)
    with pytest.raises(ValueError):
        sharp_test(np.zeros((2, 5)), np.zeros((2, 5)), experimental=True)
    with pytest.raises(ValueError):
        sharp_test(np.zeros(1), np.zeros(1), experimental=True)


def test_simulator_matches_the_assumed_covariance():
    # Guards the rest of the file: every calibration number below is only
    # meaningful if the fast sampler really draws from Sigma(sigma2, rho).
    rng = np.random.default_rng(0)
    J, sigma2, rho = 6, 2.0, 0.3
    diff_a, diff_b = _simulate_sharp_data(J, 0.5, sigma2, rho, rng, size=200000)
    sample = np.concatenate([diff_a, diff_b], axis=1)
    assert np.abs(np.cov(sample.T) - _sharp_covariance(J, sigma2, rho)).max() < 0.05
    assert sample.mean() == pytest.approx(0.5, abs=0.02)


def test_closed_form_likelihood_matches_dense_linear_algebra():
    # The estimator never builds Sigma; it uses closed forms for log|Sigma|
    # and the quadratic form derived from the sum/difference rotation. Check
    # them against literal 2J x 2J slogdet/solve, and check that the score
    # statistic equals the D^T Sigma^-1 1 / sqrt(1^T Sigma^-1 1) it stands
    # for.
    rng = np.random.default_rng(1)
    for J in (2, 5, 30):
        lo, hi = rho_bounds(J)
        assert lo == pytest.approx(-1.0 / (2 * (J - 1)))
        assert hi == 0.5
        for rho in (lo + 1e-3, -0.01, 0.0, 0.1, 0.49):
            for sigma2 in (0.5, 3.0):
                cov = _sharp_covariance(J, sigma2, rho)
                assert np.linalg.eigvalsh(cov).min() > 0, (J, rho)
                diff_a, diff_b = _simulate_sharp_data(J, 0.3, 1.0, 0.2, rng)
                d = np.concatenate([diff_a, diff_b])
                d_bar, ss_s, ss_delta, _, _ = _sufficient_stats(diff_a, diff_b)

                for mu in (0.0, 0.4):
                    resid = d - mu
                    quad_dense = resid @ np.linalg.solve(cov, resid)
                    a = 1.0 + 2.0 * rho * (J - 1)
                    b = 1.0 - 2.0 * rho
                    quad_closed = (
                        2 * J * (d_bar - mu) ** 2 / a + 0.5 * ss_s / b + 0.5 * ss_delta
                    ) / sigma2
                    assert quad_dense == pytest.approx(quad_closed, rel=1e-10)
                    logdet_closed = (
                        2 * J * np.log(sigma2) + np.log(a) + (J - 1) * np.log(b)
                    )
                    assert np.linalg.slogdet(cov)[1] == pytest.approx(
                        logdet_closed, rel=1e-10
                    )

                ones = np.ones(2 * J)
                cov_inv_1 = np.linalg.solve(cov, ones)
                z_dense = (d @ cov_inv_1) / np.sqrt(ones @ cov_inv_1)
                assert z_dense == pytest.approx(
                    d_bar / np.sqrt(var_d_bar(J, sigma2, rho)), rel=1e-10
                )


def test_restricted_mle_finds_the_global_optimum():
    # The nuisance fit is a 1-D bracketed search; make sure it lands on the
    # global optimum of the profile likelihood and not a local one or a
    # boundary, by comparing against an exhaustive scan.
    rng = np.random.default_rng(2)
    worst_gap = 0.0
    for J in (3, 10, 30, 100):
        lo, hi = rho_bounds(J)
        margin = 1e-6 * (hi - lo)
        dense = np.linspace(lo + margin, hi - margin, 200001)
        for rho_true in (0.0, 0.05, 0.2, 0.45):
            diff_a, diff_b = _simulate_sharp_data(J, 0.0, 1.0, rho_true, rng)
            d_bar, ss_s, ss_delta, s_a2, s_b2 = _sufficient_stats(diff_a, diff_b)
            _, rho_mom = _mom_estimates(J, ss_delta, s_a2, s_b2)
            # also probe hypothesized means far from D-bar, as the CI
            # inversion does
            for t in (d_bar, 0.0, d_bar + 5.0):
                _, rho_hat = _restricted_mle(J, t, ss_s, ss_delta, rho_mom)
                mine = _neg_profile_nll(rho_hat, J, t, ss_s, ss_delta)
                best = float(np.nanmin(_neg_profile_nll(dense, J, t, ss_s, ss_delta)))
                worst_gap = max(worst_gap, mine - best)
    assert worst_gap < 1e-9, worst_gap


def test_sharp_test_parameter_recovery():
    # With many repetitions the null-restricted fit should recover the true
    # simulation parameters, as should the method-of-moments values it is
    # seeded from.
    rng = np.random.default_rng(3)
    J = 2000
    for rho_true in (0.05, 0.3):
        diff_a, diff_b = _simulate_sharp_data(J, 0.0, 2.0, rho_true, rng)
        result = sharp_test(diff_a, diff_b, experimental=True)
        assert result.sigma2 == pytest.approx(2.0, abs=0.2)
        assert result.rho == pytest.approx(rho_true, abs=0.05)
        assert result.sigma2_mom == pytest.approx(2.0, abs=0.2)
        assert result.rho_mom == pytest.approx(rho_true, abs=0.05)
        assert result.n_repeats == J


def test_p_value_matches_the_public_api():
    # The Monte Carlo helpers above bypass sharp_test for speed; pin them to
    # the real entry point so the calibration tests are testing what callers
    # actually get.
    rng = np.random.default_rng(4)
    for J, rho in ((10, 0.1), (30, 0.3)):
        diff_a, diff_b = _simulate_sharp_data(J, 0.2, 1.0, rho, rng, size=5)
        for i in range(5):
            d_bar, ss_s, ss_delta, s_a2, s_b2 = _sufficient_stats(diff_a[i], diff_b[i])
            _, rho_mom = _mom_estimates(J, ss_delta, s_a2, s_b2)
            z = _score_z(J, d_bar, ss_s, ss_delta, rho_mom)[0]
            expected = 2 * stats.norm.sf(abs(z))
            got = sharp_test(diff_a[i], diff_b[i], experimental=True)
            assert got.p_value == pytest.approx(expected, rel=1e-12)


def test_confidence_interval_inverts_the_test():
    # The CI is the set of hypothesized means the test would not reject, so
    # each endpoint must sit exactly at p = alpha, and the interval must
    # contain the point estimate and be centred on it.
    rng = np.random.default_rng(5)
    alpha = 0.05
    for J, rho, mu in ((10, 0.2, 0.5), (30, 0.05, 0.0), (60, 0.35, 1.0)):
        diff_a, diff_b = _simulate_sharp_data(J, mu, 1.0, rho, rng)
        result = sharp_test(diff_a, diff_b, alpha=alpha, experimental=True)
        low, high = result.ci
        assert low < result.mean_diff < high
        assert (low + high) / 2 == pytest.approx(result.mean_diff, rel=1e-8)
        d_bar, ss_s, ss_delta, s_a2, s_b2 = _sufficient_stats(diff_a, diff_b)
        _, rho_mom = _mom_estimates(J, ss_delta, s_a2, s_b2)
        for endpoint in (low, high):
            z = _score_z(J, d_bar - endpoint, ss_s, ss_delta, rho_mom)[0]
            assert 2 * stats.norm.sf(abs(z)) == pytest.approx(alpha, rel=1e-6)
        # p < alpha and "0 outside the CI" have to be the same statement
        assert (result.p_value < alpha) is not (low <= 0.0 <= high)


def test_statistic_is_bounded_and_warns_when_it_cannot_reject():
    # |Z| plateaus near sqrt(J + 1) instead of diverging, so the p-value has
    # a floor; at J = 2 that floor sits above the usual alpha.
    rng = np.random.default_rng(6)
    for J in (5, 30):
        diff_a, diff_b = _simulate_sharp_data(J, 0.0, 1.0, 0.15, rng)
        d_bar, ss_s, ss_delta, s_a2, s_b2 = _sufficient_stats(diff_a, diff_b)
        _, rho_mom = _mom_estimates(J, ss_delta, s_a2, s_b2)
        plateau = max(
            abs(_score_z(J, t, ss_s, ss_delta, rho_mom)[0])
            for t in np.logspace(0, 3, 40)
        )
        assert plateau == pytest.approx(np.sqrt(J + 1.0), rel=0.05)

    assert smallest_meaningful_p_value(2) > 0.05
    assert smallest_meaningful_p_value(30) < 1e-6
    with pytest.warns(UserWarning, match="cannot reject"):
        sharp_test(np.array([0.0, 9.0]), np.array([1.0, 8.0]), experimental=True)


def test_false_positive_rate_is_never_inflated():
    # The headline safety property, and the one the estimator this replaced
    # did not have: across J and rho the test must not over-reject. Sampled
    # coarsely here (the dense 7 x 17 grid in the module docstring used 4000
    # draws per cell); the bound allows for Monte Carlo noise at n = 800.
    rng = np.random.default_rng(7)
    n_sims = 800
    for J in (10, 30, 100):
        for rho in (0.0, 0.1, 0.3, 0.49):
            fpr = np.mean(_p_values(J, 0.0, 1.0, rho, n_sims, rng) < 0.05)
            assert fpr < 0.085, f"J={J} rho={rho} FPR={fpr}"


def test_calibration_is_near_nominal_at_larger_correlation():
    # Where the module docstring's table says the test is usable, it should
    # actually reject at close to the nominal rate rather than being
    # uselessly conservative.
    rng = np.random.default_rng(8)
    n_sims = 1500
    for J, rho in ((30, 0.45), (100, 0.3), (300, 0.2)):
        fpr = np.mean(_p_values(J, 0.0, 1.0, rho, n_sims, rng) < 0.05)
        assert 0.025 < fpr < 0.085, f"J={J} rho={rho} FPR={fpr}"


def test_known_conservative_region():
    # Regression benchmark for the documented weak spot, which for the score
    # test is conservativeness (safe direction) rather than the previous
    # estimator's anti-conservativeness: at small true correlation the test
    # essentially never rejects, and the smaller J is the wider that region.
    # If these come back near nominal, that is good news -- update the
    # module docstring's table and this test together.
    rng = np.random.default_rng(9)
    n_sims = 1500
    for J, rho, upper in ((10, 0.05, 0.01), (30, 0.05, 0.015), (30, 0.1, 0.03)):
        fpr = np.mean(_p_values(J, 0.0, 1.0, rho, n_sims, rng) < 0.05)
        assert fpr < upper, f"J={J} rho={rho} FPR={fpr} no longer conservative"


def test_power_is_reasonable_at_larger_correlation_and_poor_at_small():
    # Power against the effect size an oracle test (one that knew
    # Var(D-bar)) would detect 80% of the time. The conservativeness above
    # is paid for here, so both directions are pinned.
    rng = np.random.default_rng(10)
    n_sims = 800
    oracle = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)

    J, rho = 30, 0.45
    mu = oracle * np.sqrt(var_d_bar(J, 1.0, rho))
    power = np.mean(_p_values(J, mu, 1.0, rho, n_sims, rng) < 0.05)
    assert power > 0.65, power

    J, rho = 30, 0.0
    mu = oracle * np.sqrt(var_d_bar(J, 1.0, rho))
    power = np.mean(_p_values(J, mu, 1.0, rho, n_sims, rng) < 0.05)
    assert power < 0.05, power


def test_sharp_compare_end_to_end():
    csv_file = os.path.join(os.path.dirname(__file__), "data", "breast_cancer.csv")
    import pandas as pd

    data = pd.read_csv(csv_file)
    X = data.iloc[:, :10].values
    y = data["target"].values

    good_clf = ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 20})
    bad_clf = ("sklearn.dummy", "DummyClassifier", {"strategy": "most_frequent"})

    result = sharp_compare(
        X,
        y,
        good_clf,
        bad_clf,
        metric="accuracy_score",
        n_repeats=20,
        n_folds=5,
        random_state=0,
        experimental=True,
    )
    # A real classifier should clearly out-perform a majority-class dummy.
    assert result.mean_diff > 0
    assert result.p_value < 0.05
    assert result.ci[0] > 0
