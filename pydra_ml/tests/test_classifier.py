import os

import numpy as np

from ..classifier import gen_workflow, run_workflow


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
        "permutation_importance_scoring": "balanced_accuracy",
        "gen_shap": True,
        "nsamples": 15,
        "l1_reg": "aic",
        "plot_top_n_shap": 16,
        "metrics": ["balanced_accuracy_score", "roc_auc_score"],
    }
    spec = gen_workflow(inputs, cache_dir=tmpdir)
    result = run_workflow(spec, "debug", {})
    # 4 outer combinations (outer-product split): (MLP,True), (MLP,False), (Pipeline,True), (Pipeline,False)
    # score[i] is a list of per-split metric lists, score[i][j][k] = combination i, split j, metric k
    permuted_ba = result.outputs.score[0][0][0]
    real_ba = result.outputs.score[1][0][0]
    assert permuted_ba < real_ba
    # MLP non-permuted final model (combination 1) should be a fitted pipeline
    assert hasattr(result.outputs.model[1], "predict")
    assert isinstance(result.outputs.model[1].predict(np.ones((1, 10))), np.ndarray)


def test_classifier_imbalanced(tmpdir):
    pytest = __import__("pytest")
    try:
        import imblearn  # noqa: F401
    except ImportError:
        pytest.skip("imbalanced-learn not installed")

    clfs = [
        ("sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}),
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
        "balancing": ["imblearn.over_sampling", "SMOTE", {"random_state": 0}],
        "bootstrap_strategy": "stratified",
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "balanced_accuracy",
        "gen_shap": False,
        "nsamples": 15,
        "l1_reg": "aic",
        "plot_top_n_shap": 16,
        "metrics": ["balanced_accuracy_score", "roc_auc_score"],
    }
    spec = gen_workflow(inputs, cache_dir=tmpdir)
    result = run_workflow(spec, "debug", {})
    # 2 outer combinations: (MLP,True), (MLP,False)
    permuted_ba = result.outputs.score[0][0][0]
    real_ba = result.outputs.score[1][0][0]
    assert real_ba > 0.5, f"balanced accuracy should be above chance, got {real_ba}"
    assert permuted_ba < real_ba


def test_regressor_imbalanced(tmpdir):
    pytest = __import__("pytest")
    try:
        import imblearn  # noqa: F401
    except ImportError:
        pytest.skip("imbalanced-learn not installed")

    clfs = [
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
        "balancing": [
            "imblearn.under_sampling",
            "RandomUnderSampler",
            {"random_state": 0},
        ],
        "balancing_bins": 5,
        "bootstrap_strategy": "stratified_regression",
        "n_bins": 5,
        "gen_feature_importance": False,
        "gen_permutation_importance": False,
        "permutation_importance_n_repeats": 5,
        "permutation_importance_scoring": "r2",
        "gen_shap": False,
        "nsamples": 15,
        "l1_reg": "aic",
        "plot_top_n_shap": 10,
        "metrics": ["explained_variance_score"],
    }
    spec = gen_workflow(inputs, cache_dir=tmpdir)
    result = run_workflow(spec, "debug", {})
    # 2 outer combinations: (LinearRegression,True), (LinearRegression,False)
    permuted_ev = result.outputs.score[0][0][0]
    real_ev = result.outputs.score[1][0][0]
    assert (
        real_ev > permuted_ev
    ), f"real EV {real_ev:.3f} should exceed permuted {permuted_ev:.3f}"
    assert hasattr(result.outputs.model[1], "predict")


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
