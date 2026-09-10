import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score

from ..report import gen_report_shap_regres, shaps_to_summary


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
