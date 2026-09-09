import os

import numpy as np
import pytest

from ..sharp_test import sharp_compare, sharp_test


def _simulate_sharp_data(J, mu, sigma2, rho, rng):
    """Draw one (diff_a, diff_b) realization directly from the covariance
    structure SHARP assumes: Cov(D_Aj, D_Bj) = 0 (same iteration, disjoint
    halves), Cov(anything else) = rho * sigma2, Var(each entry) = sigma2.
    """
    n = 2 * J
    cov = np.full((n, n), rho * sigma2)
    np.fill_diagonal(cov, sigma2)
    for j in range(J):
        cov[j, J + j] = 0.0
        cov[J + j, j] = 0.0
    x = rng.multivariate_normal(np.full(n, mu), cov)
    return x[:J], x[J:]


def test_sharp_test_input_validation():
    with pytest.raises(ValueError):
        sharp_test(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError):
        sharp_test(np.zeros(1), np.zeros(1))


def test_sharp_test_parameter_recovery():
    # With a large number of repetitions and a moderate correlation (away
    # from the rho=0 boundary region -- see the module docstring caveat),
    # the fitted sigma2/rho should recover the true simulation parameters.
    rng = np.random.RandomState(0)
    J = 2000
    diff_a, diff_b = _simulate_sharp_data(J, mu=0.0, sigma2=2.0, rho=0.3, rng=rng)
    result = sharp_test(diff_a, diff_b)
    assert result.sigma2 == pytest.approx(2.0, abs=0.2)
    assert result.rho == pytest.approx(0.3, abs=0.05)
    assert result.sigma2_null == pytest.approx(2.0, abs=0.2)
    assert result.rho_null == pytest.approx(0.3, abs=0.05)


def test_sharp_test_calibration_and_power_at_moderate_correlation():
    # Caveat (see module docstring): the null-restricted MLE used for the
    # score test is conservative when the true between-repetition
    # correlation is near its structural lower bound (rho close to 0).
    # At a moderate, more realistic correlation, false positive control and
    # power both behave as expected: FPR near nominal alpha, and high power
    # for a clearly real difference.
    rng = np.random.RandomState(1)
    J, rho, sigma2 = 30, 0.3, 1.0

    n_sims = 300
    false_positives = sum(
        sharp_test(*_simulate_sharp_data(J, 0.0, sigma2, rho, rng)).p_value < 0.05
        for _ in range(n_sims)
    )
    # Allow some margin above the nominal 0.05 for simulation noise; the
    # important property is that it is not wildly anti-conservative.
    assert false_positives / n_sims < 0.12

    # With Var(D_bar) = 1 * (1/60 + 29/30 * 0.3) ~= 0.307 at this (J, rho,
    # sigma2), mu=2.0 gives an effect/sd ratio of ~3.6, i.e. a well-
    # calibrated two-sided z-test would have ~95% power here.
    n_sims = 200
    true_positives = sum(
        sharp_test(*_simulate_sharp_data(J, 2.0, sigma2, rho, rng)).p_value < 0.05
        for _ in range(n_sims)
    )
    assert true_positives / n_sims > 0.8


def test_sharp_test_never_wildly_anticonservative():
    # Even in the conservative corner (rho near 0), false positive control
    # must never be violated in the unsafe direction (FPR >> alpha).
    rng = np.random.RandomState(2)
    n_sims = 300
    for J, rho in [(10, 0.0), (30, 0.0), (60, 0.1)]:
        false_positives = sum(
            sharp_test(*_simulate_sharp_data(J, 0.0, 1.0, rho, rng)).p_value < 0.05
            for _ in range(n_sims)
        )
        assert false_positives / n_sims < 0.1, f"J={J} rho={rho}"


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
        metric="accuracy",
        n_repeats=20,
        n_folds=5,
        random_state=0,
    )
    # A real classifier should clearly out-perform a majority-class dummy.
    assert result.mean_diff > 0
    assert result.p_value < 0.05
