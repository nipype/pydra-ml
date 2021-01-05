import os
from ..classifier import gen_workflow, run_workflow
import numpy as np


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
        "x_indices": [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "mean compactness",
            "mean concavity",
            "mean concave points",
            "mean symmetry",
            "mean fractal dimension",
            "radius error",
            "texture error",
            "perimeter error",
            "area error",
            "smoothness error",
            "compactness error",
            "concavity error",
            "concave points error",
            "symmetry error",
            "fractal dimension error",
            "worst radius",
            "worst texture",
            "worst perimeter",
            "worst area",
            "worst smoothness",
            "worst compactness",
            "worst concavity",
            "worst concave points",
            "worst symmetry",
            "worst fractal dimension",
        ],
        "target_vars": ("target",),
        "group_var": None,
        "n_splits": 2,
        "test_size": 0.2,
        "clf_info": clfs,
        "permute": [True, False],
        "gen_shap": True,
        "nsamples": 5,
        "l1_reg": "aic",
        "plot_top_n_shap": 16,
        "metrics": ["roc_auc_score", "accuracy_score"],
    }
    tmpdir.chdir()
    wf = gen_workflow(inputs, cache_dir=tmpdir / "cache")
    results = run_workflow(wf, "cf", {"n_procs": 1})
    assert results[0][0]["ml_wf.clf_info"][1] == "MLPClassifier"
    assert results[0][0]["ml_wf.permute"]
    assert results[0][1].output.score[0][0] < results[1][1].output.score[0][0]
    assert hasattr(results[2][1].output.model, "predict")
    assert isinstance(results[2][1].output.model.predict(np.ones((1, 30))), np.ndarray)
    assert all([val[1].output.model.n_features_in_ == 30 for val in results])


def test_regressor(tmpdir):
    clfs = [
        [
            ["sklearn.impute", "SimpleImputer"],
            ["sklearn.preprocessing", "StandardScaler"],
            ["sklearn.neural_network", "MLPRegressor", {"alpha": 1, "max_iter": 1000}],
        ],
        (
            "sklearn.linear_model",
            "LinearRegression",
            {"fit_intercept": True, "normalize": True},
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
        "gen_shap": True,
        "nsamples": 5,
        "l1_reg": "aic",
        "plot_top_n_shap": 10,
        "metrics": ["explained_variance_score"],
    }

    tmpdir.chdir()
    wf = gen_workflow(inputs, cache_dir=tmpdir / "cache")
    results = run_workflow(wf, "cf", {"n_procs": 1})
    assert results[0][0]["ml_wf.clf_info"][-1][1] == "MLPRegressor"
    assert results[0][0]["ml_wf.permute"]
    assert results[0][1].output.score[0][0] < results[1][1].output.score[0][0]
    assert hasattr(results[2][1].output.model, "predict")
    assert isinstance(results[2][1].output.model.predict(np.ones((1, 10))), np.ndarray)
    assert all([val[1].output.model.n_features_in_ == 10 for val in results])
