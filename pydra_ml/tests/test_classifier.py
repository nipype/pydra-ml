import os
from ..classifier import gen_workflow, run_workflow

clfs = [
    ("sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}),
    ("sklearn.naive_bayes", "GaussianNB", {}),
]
csv_file = os.path.join(os.path.dirname(__file__), "data", "breast_cancer.csv")
inputs = {
    "filename": csv_file,
    "x_indices": range(30),
    "target_vars": ("target",),
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


def test_classifier(tmpdir):
    wf = gen_workflow(inputs, cache_dir=tmpdir)
    results = run_workflow(wf, "cf", {"n_procs": 1})
    assert results[0][0]["ml_wf.clf_info"][1] == "MLPClassifier"
    assert results[0][0]["ml_wf.permute"]
    assert results[0][1].output.score[0][0] < results[1][1].output.score[0][0]
