import os

import numpy as np
import pytest

from ..classifier import gen_workflow, run_workflow
from ..tasks import (
    _fit_with_optional_target_weights,
    _target_sample_weights,
    gen_splits,
    get_permutation_importance,
    read_file,
    train_test_kernel,
)


def test_classifier(tmpdir):
    clfs = [
        ("sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}),
        [
            ["sklearn.impute", "SimpleImputer"],
            ["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.naive_bayes", "GaussianNB", {}],
        ],
    ]
    csv_file = os.path.join(os.path.dirname(__file__), "data", "breast_cancer.csv")
    inputs = {
        "filename": csv_file,
        "x_indices": list(range(10)),
        "target_vars": ("target",),
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [True, False],
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "accuracy",
        "gen_shap": True,
        "nsamples": 15,
        "l1_reg": "aic",
        "plot_top_n_shap": 16,
        "metrics": ["roc_auc_score", "accuracy_score"],
    }
    spec = gen_workflow(inputs, cache_dir=tmpdir)
    result = run_workflow(spec, "debug", {})
    # 4 outer combinations (outer-product split): (MLP,True), (MLP,False), (Pipeline,True), (Pipeline,False)
    # score[i] is a list of per-split metric lists, score[i][j][k] = combination i, split j, metric k
    permuted_auc = result.outputs.score[0][0][0]
    real_auc = result.outputs.score[1][0][0]
    assert permuted_auc < real_auc
    # MLP non-permuted final model (combination 1) should be a fitted pipeline
    assert hasattr(result.outputs.model[1], "predict")
    assert isinstance(result.outputs.model[1].predict(np.ones((1, 10))), np.ndarray)


def test_get_permutation_importance_uses_full_pipeline():
    # Regression test: get_permutation_importance used to score just the
    # final estimator (pipe.steps[-1][1]) on raw, unscaled X, bypassing the
    # StandardScaler step -- the estimator never saw data on that scale
    # during training, so it scored near chance and returned near-zero
    # importances for every feature, silently.
    csv_file = os.path.join(os.path.dirname(__file__), "data", "breast_cancer.csv")
    X, y, groups, feature_names = read_file(
        csv_file, x_indices=list(range(10)), target_vars=("target",)
    )
    splits, split_indices, _ = gen_splits(2, 0.2, X, y, groups)
    clf_info = ("sklearn.linear_model", "LogisticRegression", {"max_iter": 1000})
    _, model = train_test_kernel(X, y, splits, 0, clf_info, permute=False)
    importances = get_permutation_importance(
        X,
        y,
        permute=False,
        model=model,
        permutation_importance_n_repeats=10,
        permutation_importance_scoring="accuracy",
        gen_permutation_importance=True,
    )
    assert np.max(np.abs(importances)) > 0.01


def test_classifier_with_resampling(tmpdir):
    # imbalanced-learn samplers (e.g. RandomOverSampler) only implement
    # fit_resample, not transform, so they can only be used as pipeline
    # steps if the pipeline is imblearn's (a superset of sklearn's).
    clfs = [
        [
            ["imblearn.over_sampling", "RandomOverSampler"],
            ["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 10}],
        ],
    ]
    csv_file = os.path.join(os.path.dirname(__file__), "data", "breast_cancer.csv")
    inputs = {
        "filename": csv_file,
        "x_indices": list(range(10)),
        "target_vars": ("target",),
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [False],
        "gen_feature_importance": True,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "accuracy",
        "gen_shap": False,
        "nsamples": 15,
        "l1_reg": "aic",
        "plot_top_n_shap": 16,
        "metrics": ["roc_auc_score", "accuracy_score"],
    }
    spec = gen_workflow(inputs, cache_dir=tmpdir)
    result = run_workflow(spec, "debug", {})
    assert hasattr(result.outputs.model[0], "predict")
    assert isinstance(result.outputs.model[0].predict(np.ones((1, 10))), np.ndarray)
    # steps[-1] must resolve to the RandomForestClassifier, not the
    # StandardScaler, now that the pipeline has a sampler step in front.
    feature_importance = result.outputs.feature_importance[0][0]
    assert len(feature_importance) == 10


def test_target_sample_weights():
    # Discrete target: exact inverse class-frequency weighting.
    y_discrete = np.array([0] * 8 + [1] * 2)
    weights = _target_sample_weights(y_discrete, n_bins=10)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[y_discrete == 1][0] / weights[y_discrete == 0][0], 4.0)

    # Continuous target: a handful of extreme values are rarer than the
    # densely-populated middle, so they should end up upweighted.
    rng = np.random.RandomState(0)
    y_continuous = np.concatenate([rng.normal(0, 1, 95), [50.0] * 5])
    weights = _target_sample_weights(y_continuous, n_bins=10)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] > weights[0]


def test_fit_with_target_weights_falls_back_when_unsupported():
    # An estimator whose fit() doesn't accept sample_weight must still get
    # fit (unweighted, with a warning) rather than raising.
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.pipeline import Pipeline

    class NoSampleWeightRegressor(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            self.mean_ = y.mean()
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    pipe = Pipeline([("reg", NoSampleWeightRegressor())])
    with pytest.warns(UserWarning, match="does not accept sample_weight"):
        _fit_with_optional_target_weights(
            pipe, X, y, balance_target=True, target_n_bins=5
        )
    assert hasattr(pipe, "predict")
    assert pipe.predict(X).shape == (20,)


def test_regressor_with_target_balancing(tmpdir):
    clfs = [
        ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 10}),
    ]
    csv_file = os.path.join(os.path.dirname(__file__), "data", "diabetes_table.csv")
    inputs = {
        "filename": csv_file,
        "x_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target_vars": ["target"],
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [False],
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "accuracy",
        "gen_shap": False,
        "nsamples": 15,
        "l1_reg": "aic",
        "plot_top_n_shap": 10,
        "metrics": ["explained_variance_score"],
        "balance_target": True,
        "target_n_bins": 5,
    }
    spec = gen_workflow(inputs, cache_dir=tmpdir)
    result = run_workflow(spec, "debug", {})
    assert hasattr(result.outputs.model[0], "predict")
    assert isinstance(result.outputs.model[0].predict(np.ones((1, 10))), np.ndarray)


def test_regressor(tmpdir):
    clfs = [
        [
            ["sklearn.impute", "SimpleImputer"],
            ["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.neural_network", "MLPRegressor", {"alpha": 1, "max_iter": 100}],
        ],
        (
            "sklearn.linear_model",
            "LinearRegression",
            {"fit_intercept": True},
        ),
    ]
    csv_file = os.path.join(os.path.dirname(__file__), "data", "diabetes_table.csv")
    inputs = {
        "filename": csv_file,
        "x_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target_vars": ["target"],
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [True, False],
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "accuracy",
        "gen_shap": True,
        "nsamples": 15,
        "l1_reg": "aic",
        "plot_top_n_shap": 10,
        "metrics": ["explained_variance_score"],
    }

    spec = gen_workflow(inputs, cache_dir=tmpdir)
    result = run_workflow(spec, "debug", {})
    # 4 outer combinations: (Pipeline,True), (Pipeline,False), (LinearRegression,True), (LinearRegression,False)
    permuted_ev = result.outputs.score[0][0][0]
    real_ev = result.outputs.score[1][0][0]
    assert permuted_ev < real_ev
    # Pipeline non-permuted final model (combination 1)
    assert hasattr(result.outputs.model[1], "predict")
    assert isinstance(result.outputs.model[1].predict(np.ones((1, 10))), np.ndarray)
