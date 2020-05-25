#!/usr/bin/env python

import pydra
import typing as ty
import numpy as np


@pydra.mark.task
@pydra.mark.annotate(
    {"return": {"X": ty.Any, "Y": ty.Any, "groups": ty.Any, "feature_names": ty.Any}}
)
def read_file(filename, x_indices=None, target_vars=None, group="groups"):
    import pandas as pd

    data = pd.read_csv(filename)
    if isinstance(x_indices[0], int):
        X = data.iloc[:, x_indices]
    elif isinstance(x_indices[0], str):
        X = data[x_indices]
    else:
        raise ValueError(f"{x_indices} is not a list of string or ints")
    Y = data[target_vars]
    if group in data.keys():
        groups = data[:, [group]]
    else:
        groups = list(range(X.shape[0]))
    feature_names = list(X.columns)
    return X.values, Y.values, groups, feature_names


@pydra.mark.task
@pydra.mark.annotate({"return": {"splits": ty.Any, "split_indices": ty.Any}})
def gen_splits(n_splits, test_size, X, Y, groups=None, random_state=0):
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    train_test_splits = list(gss.split(X, Y, groups=groups))
    split_indices = list(range(n_splits))
    return train_test_splits, split_indices


@pydra.mark.task
@pydra.mark.annotate({"return": {"output": ty.Any, "model": ty.Any}})
def train_test_kernel(X, y, train_test_split, split_index, clf_info, permute, metrics):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np

    mod = __import__(clf_info[0], fromlist=[clf_info[1]])
    params = {}
    if len(clf_info) > 2:
        params = clf_info[2]
    clf = getattr(mod, clf_info[1])(**params)
    if len(clf_info) == 4:
        from sklearn.model_selection import GridSearchCV

        clf = GridSearchCV(clf, param_grid=clf_info[3])
    train_index, test_index = train_test_split[split_index]
    pipe = Pipeline([("std", StandardScaler()), (clf_info[1], clf)])
    y = y.ravel()
    if permute:
        pipe.fit(X[train_index], y[np.random.permutation(train_index)])
    else:
        pipe.fit(X[train_index], y[train_index])
    predicted = pipe.predict(X[test_index])
    return (y[test_index], predicted), (pipe, train_index, test_index)


@pydra.mark.task
@pydra.mark.annotate({"return": {"score": ty.Any, "output": ty.Any}})
def calc_metric(output, metrics):
    score = []
    for metric in metrics:
        metric_mod = __import__("sklearn.metrics", fromlist=[metric])
        metric_func = getattr(metric_mod, metric)
        score.append(metric_func(output[0], output[1]))
    return score, output


@pydra.mark.task
@pydra.mark.annotate({"return": {"shaps": ty.Any}})
def get_shap(X, permute, model, gen_shap=False, nsamples="auto", l1_reg="aic"):
    if permute or not gen_shap:
        return []
    pipe, train_index, test_index = model
    import shap

    explainer = shap.KernelExplainer(pipe.predict, shap.kmeans(X[train_index], 5))
    shaps = explainer.shap_values(X[test_index], nsamples=nsamples, l1_reg=l1_reg)
    return shaps
