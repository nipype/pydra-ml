#!/usr/bin/env python

import pandas as pd
import numpy as np
import pydra
import typing as ty


@pydra.mark.task
@pydra.mark.annotate({"return": {"X": ty.Any, "Y": ty.Any, "groups": ty.Any}})
def read_file(filename, x_indices=None, target_vars=None, group='groups'):
    data = pd.read_csv(filename)
    X = data.iloc[:, x_indices]
    Y = data[target_vars]
    if group in data.keys():
        groups = data[:, [group]]
    else:
        groups = list(range(X.shape[0]))
    return X.values, Y.values, groups


@pydra.mark.task
@pydra.mark.annotate({"return": {"splits": ty.Any, "split_indices": ty.Any}})
def gen_splits(n_splits, test_size, X, Y, groups=None, random_state=0):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                            random_state=random_state)
    train_test_splits = list(gss.split(X, Y, groups=groups))
    split_indices = list(range(n_splits))
    return train_test_splits, split_indices


@pydra.mark.task
@pydra.mark.annotate({"return": {"auc": ty.Any,
                                 "output": ty.Any,
                                 "model": ty.Any}})
def train_test_kernel(X, y, train_test_split, split_index, clf_info, permute):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    mod = __import__(clf_info[0], fromlist=[clf_info[1]])
    clf = getattr(mod, clf_info[1])(**clf_info[2])
    if len(clf_info) > 3:
        from sklearn.model_selection import GridSearchCV
        clf = GridSearchCV(clf, param_grid=clf_info[3])
    train_index, test_index = train_test_split[split_index]
    pipe = Pipeline([('std', StandardScaler()), (clf_info[1], clf)])
    y = y.ravel()
    if permute:
        pipe.fit(X[train_index], y[np.random.permutation(train_index)])
    else:
        pipe.fit(X[train_index], y[train_index])

    predicted = pipe.predict(X[test_index])
    auc = roc_auc_score(y[test_index], predicted)
    return auc, (y[test_index], predicted), pipe


@pydra.mark.task
@pydra.mark.annotate({"return": {"shaps": ty.Any}})
def get_shap(X, train_test_split, split_index, permute, model, noshap=False,
             nsamples="auto", l1_reg="aic"):
    if permute or noshap:
        return []
    train_index, test_index = train_test_split[split_index]
    import shap
    explainer = shap.KernelExplainer(model.predict,
                                     shap.kmeans(X[train_index], 5))
    shaps = explainer.shap_values(X[test_index], nsamples=nsamples,
                                  l1_reg=l1_reg)
    return shaps
