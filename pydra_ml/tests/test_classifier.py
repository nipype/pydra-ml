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
    "noshap": False,
    "nsamples": 5,
    "l1_reg": "aic",
}
n_procs = 1


def test_classifier(tmpdir):
    cache_dir = tmpdir
    wf = gen_workflow(inputs, cache_dir=cache_dir)
    results = run_workflow(wf, n_procs)
    assert results is not None
