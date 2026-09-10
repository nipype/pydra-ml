import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import explained_variance_score

from ..report import (
    _clf_labels,
    compute_pairwise_stats,
    corrected_resampled_ttest,
    gen_report_shap_regres,
    shaps_to_summary,
)


def test_shaps_to_summary_stats_use_only_split_columns(tmpdir):
    # Regression test: mean/std/min/max must be computed from the raw
    # per-split values only, not from each other (e.g. "max" must not
    # include the already-appended "mean"/"std" columns).
    data = {0: [1.0, 100.0], 1: [2.0, 200.0], 2: [3.0, 300.0]}
    shaps_n_splits = pd.DataFrame(data)
    expected_max = shaps_n_splits.max(axis=1).copy()
    expected_min = shaps_n_splits.min(axis=1).copy()
    expected_std = shaps_n_splits.std(axis=1).copy()

    shaps_to_summary(
        shaps_n_splits,
        feature_names=["f0", "f1"],
        output_dir=str(tmpdir) + "/",
        filename="test",
        plot_top_n_shap=1.0,
    )

    result = pd.read_csv(str(tmpdir) + "/summary_values_test.csv", index_col=0)
    assert np.allclose(sorted(result["max"]), sorted(expected_max))
    assert np.allclose(sorted(result["min"]), sorted(expected_min))
    assert np.allclose(sorted(result["std"]), sorted(expected_std))


def test_corrected_resampled_ttest_no_difference():
    t_stat, p_value = corrected_resampled_ttest(np.zeros(10), test_train_ratio=0.25)
    assert t_stat == 0.0
    assert p_value == 1.0


def test_corrected_resampled_ttest_wider_than_naive():
    # The whole point of the correction: it must inflate the variance (and
    # so the p-value) relative to a naive paired t-test on the same diffs,
    # since splits share overlapping data and a naive test assumes they
    # don't.
    rng = np.random.RandomState(0)
    diffs = rng.normal(loc=0.01, scale=0.02, size=30)
    _, p_corrected = corrected_resampled_ttest(diffs, test_train_ratio=0.25)
    from scipy import stats as scipy_stats

    _, p_naive = scipy_stats.ttest_1samp(diffs, 0.0)
    assert p_corrected > p_naive


def test_corrected_resampled_ttest_detects_clear_difference():
    diffs = np.full(20, 0.05) + np.random.RandomState(1).normal(0, 0.001, 20)
    _, p_value = corrected_resampled_ttest(diffs, test_train_ratio=0.25)
    assert p_value < 0.01


def test_corrected_resampled_ttest_matches_hand_computation():
    # Pin the exact formula (variance correction, ddof, df, two-sidedness)
    # against an independent hand computation, rather than only checking
    # directional properties that many different (wrong) formulas satisfy.
    diffs = np.array([0.1, -0.05, 0.2, 0.0, 0.05])
    ratio = 0.3
    n = len(diffs)
    expected_var = (1.0 / n + ratio) * diffs.var(ddof=1)
    expected_t = diffs.mean() / np.sqrt(expected_var)
    from scipy import stats as scipy_stats

    expected_p = 2 * scipy_stats.t.sf(np.abs(expected_t), df=n - 1)

    t_stat, p_value = corrected_resampled_ttest(diffs, test_train_ratio=ratio)
    assert np.isclose(t_stat, expected_t)
    assert np.isclose(p_value, expected_p)


def test_corrected_resampled_ttest_detects_significance_naive_would_miss():
    # The actual scientific claim behind the correction: for a genuine
    # between-split-correlated design, some differences the naive
    # (uncorrected) paired t-test calls significant should not be, once the
    # overlapping-split correlation is accounted for.
    rng = np.random.RandomState(3)
    diffs = rng.normal(loc=0.02, scale=0.03, size=15)
    from scipy import stats as scipy_stats

    _, p_naive = scipy_stats.ttest_1samp(diffs, 0.0)
    _, p_corrected = corrected_resampled_ttest(diffs, test_train_ratio=0.25)
    assert p_naive < 0.05
    assert p_corrected > p_naive


def test_corrected_resampled_ttest_requires_valid_ratio():
    with pytest.raises(ValueError):
        corrected_resampled_ttest(np.array([0.1, 0.2, 0.3]), test_train_ratio=0)
    with pytest.raises(ValueError):
        corrected_resampled_ttest(np.array([0.1, 0.2, 0.3]), test_train_ratio=-0.25)


def test_corrected_resampled_ttest_handles_too_few_diffs():
    assert corrected_resampled_ttest(np.array([0.1]), test_train_ratio=0.25) == (
        pytest.approx(np.nan, nan_ok=True),
        pytest.approx(np.nan, nan_ok=True),
    )
    assert corrected_resampled_ttest(np.array([]), test_train_ratio=0.25) == (
        pytest.approx(np.nan, nan_ok=True),
        pytest.approx(np.nan, nan_ok=True),
    )


def test_corrected_resampled_ttest_zero_variance_uses_sign_test_not_certainty():
    # All diffs identical and nonzero: the old code asserted p=0 (impossible
    # certainty from a finite sample). It must instead degrade to the exact
    # two-sided sign-test p-value, which is never exactly zero.
    diffs = np.full(6, 0.03)
    t_stat, p_value = corrected_resampled_ttest(diffs, test_train_ratio=0.25)
    assert p_value == pytest.approx(2.0**-5)
    assert t_stat > 0


def test_corrected_resampled_ttest_ignores_nonfinite_diffs():
    diffs = np.array([0.05, np.nan, 0.06, 0.04, np.nan, 0.05])
    t_stat, p_value = corrected_resampled_ttest(diffs, test_train_ratio=0.25)
    finite_diffs = np.array([0.05, 0.06, 0.04, 0.05])
    expected_t, expected_p = corrected_resampled_ttest(
        finite_diffs, test_train_ratio=0.25
    )
    assert np.isclose(t_stat, expected_t)
    assert np.isclose(p_value, expected_p)


def _paired_df(n_splits, scores_a, scores_b, order_b="forward"):
    b_splits = list(range(n_splits))
    if order_b == "reversed":
        b_splits = list(reversed(b_splits))
    rows = []
    for split in range(n_splits):
        rows.append(
            {
                "Classifier": "A",
                "type": "data",
                "score": scores_a[split],
                "split": split,
            }
        )
        rows.append({"Classifier": "A", "type": "null", "score": 0.5, "split": split})
    for split in b_splits:
        rows.append(
            {
                "Classifier": "B",
                "type": "data",
                "score": scores_b[split],
                "split": split,
            }
        )
        rows.append({"Classifier": "B", "type": "null", "score": 0.5, "split": split})
    return pd.DataFrame(rows)


def test_compute_pairwise_stats_detects_paired_difference_masked_by_variance():
    # Reproduces the bug this replaces: comparing a model's mean score to
    # another model's raw score *distribution* misses a small, consistent,
    # real difference when between-split variance is large -- the paired
    # test on shared splits (via the "split" column) should not.
    rng = np.random.RandomState(0)
    n_splits = 20
    base = rng.uniform(0.5, 0.95, size=n_splits)  # large, shared between-split noise
    # A small amount of jitter is added so the diffs aren't degenerately
    # identical (which would take the zero-variance sign-test branch
    # instead of exercising the t-test this is meant to test).
    jitter = rng.normal(0, 0.002, size=n_splits)
    scores_a = base + 0.02 + jitter  # A is consistently ~0.02 better than B
    scores_b = base

    df = _paired_df(n_splits, scores_a, scores_b)
    effects, pvalues, adjusted_pvalues = compute_pairwise_stats(
        df, test_train_ratio=0.25
    )
    order = [g[0] for g in df.groupby("Classifier")]
    idx_a, idx_b = order.index("A"), order.index("B")
    assert pvalues[idx_a, idx_b] < 0.01
    assert adjusted_pvalues[idx_a, idx_b] < 0.01
    assert effects[idx_a, idx_b] > 0  # A scores higher than B


def test_compute_pairwise_stats_pairs_by_split_not_position():
    # The whole point of the "split" column: two classifiers' scores must be
    # differenced split-by-split, not by row position. Feeding B's rows in
    # reversed split order must give the identical result to in-order rows.
    rng = np.random.RandomState(0)
    n_splits = 10
    base = rng.uniform(0.5, 0.95, size=n_splits)
    scores_a = base + 0.02
    scores_b = base

    df_ordered = _paired_df(n_splits, scores_a, scores_b, order_b="forward")
    df_reversed = _paired_df(n_splits, scores_a, scores_b, order_b="reversed")

    _, pvalues_ordered, _ = compute_pairwise_stats(df_ordered, test_train_ratio=0.25)
    _, pvalues_reversed, _ = compute_pairwise_stats(df_reversed, test_train_ratio=0.25)
    assert np.allclose(pvalues_ordered, pvalues_reversed, equal_nan=True)


def test_compute_pairwise_stats_rejects_duplicate_split_per_classifier():
    # A defensive backstop for the classifier-name-collision bug _clf_labels
    # exists to prevent: two "data" rows for the same classifier on the same
    # split silently turn a 1:1 pairing into a many-to-many join.
    df = pd.DataFrame(
        [
            {"Classifier": "A", "type": "data", "score": 0.8, "split": 0},
            {"Classifier": "A", "type": "data", "score": 0.9, "split": 0},
            {"Classifier": "B", "type": "data", "score": 0.7, "split": 0},
        ]
    )
    with pytest.raises(ValueError, match="more than one 'data' row"):
        compute_pairwise_stats(df, test_train_ratio=0.25)


def test_compute_pairwise_stats_flips_direction_for_lower_is_better_metric():
    # brier_score_loss (lower is better) must not be treated as
    # higher-is-better just because its column is also named "score".
    n_splits = 10
    rng = np.random.RandomState(0)
    base = rng.uniform(0.1, 0.3, size=n_splits)
    scores_a = base  # A has the *lower* (better) brier loss
    scores_b = base + 0.05

    df = _paired_df(n_splits, scores_a, scores_b)
    effects, pvalues, _ = compute_pairwise_stats(
        df, test_train_ratio=0.25, metric_name="brier_score_loss"
    )
    order = [g[0] for g in df.groupby("Classifier")]
    idx_a, idx_b = order.index("A"), order.index("B")
    # A is the better model here, so its effect vs B must be positive even
    # though its raw score is numerically lower.
    assert effects[idx_a, idx_b] > 0


def test_clf_labels_disambiguates_same_base_name():
    # Two different RandomForestClassifier hyperparameterizations reduce to
    # the same base name ("RandomForest") -- they must not collide into one
    # report row/column.
    clf_a = ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 10})
    clf_b = ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100})
    clf_c = ("sklearn.linear_model", "LogisticRegression")
    labels = _clf_labels([clf_a, clf_b, clf_c, clf_a])

    assert labels[repr(clf_a)] != labels[repr(clf_b)]
    assert labels[repr(clf_a)].startswith("RandomForest-")
    assert labels[repr(clf_b)].startswith("RandomForest-")
    # A non-colliding base name is left untouched.
    assert labels[repr(clf_c)] == "LogisticRegression"
    # The same clf_info repeated (e.g. once per permute value) keeps one
    # consistent label rather than being treated as a second collision.
    assert len(labels) == 3


class _Output:
    pass


class _Result:
    output = None


def _make_regression_shap_result(feature_names, shaps_per_split, y_true_preds):
    r = _Result()
    r.output = _Output()
    r.output.feature_names = feature_names
    r.output.shaps = shaps_per_split
    r.output.output = y_true_preds
    params = {"ml_wf.clf_info": ("sklearn.linear_model", "LinearRegression")}
    return [(params, r)]


def test_regression_shap_weighting_does_not_invert_ranking_for_poor_models(tmpdir):
    # Reproduces the bug where explained_variance_score's unbounded lower
    # range (for poor-fitting splits) flipped the sign of, and so inverted
    # the ranking of, the performance-weighted SHAP importances.
    feature_names = ["f_tiny", "f_small", "f_med", "f_big"]
    true_importance = np.array([1.0, 2.0, 3.0, 10.0])
    n_predictions = 4

    # One good-fit split (high EV) and two badly-fit splits (very negative
    # EV), so an unclipped weight would be dominated by the negative splits.
    y_good = np.array([1.0, 2.0, 3.0, 4.0])
    y_bad = np.array([-100.0, 200.0, -300.0, 400.0])
    splits = [(y_good, y_good), (y_good, y_bad), (y_good, -y_bad)]
    evs = [explained_variance_score(yt, yp) for yt, yp in splits]
    assert evs[0] > 0 and evs[1] < -1 and evs[2] < -1, evs

    shaps_per_split = [np.tile(true_importance, (n_predictions, 1)) for _ in splits]
    result = _make_regression_shap_result(feature_names, shaps_per_split, splits)

    output_dir = str(tmpdir) + "/"
    gen_report_shap_regres(result, output_dir=output_dir, plot_top_n_shap=1.0)

    shap_dirs = [d for d in tmpdir.listdir() if d.basename.startswith("shap-")]
    assert len(shap_dirs) == 1
    summary = pd.read_csv(
        str(
            shap_dirs[0].join(
                "summary_values_shap_LinearRegression_all_predictions.csv"
            )
        ),
        index_col=0,
    )
    # The most truly important feature must rank first (largest mean), not
    # last -- clipping the weight to [0, 1] prevents the negative-EV splits
    # from dominating and reversing the order.
    ranked = list(summary.sort_values("mean", ascending=False).index)
    assert ranked == ["f_big", "f_med", "f_small", "f_tiny"]
