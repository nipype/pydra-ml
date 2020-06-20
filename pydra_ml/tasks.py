#!/usr/bin/env python


def read_file(filename, x_indices=None, target_vars=None, group=None):
    """Read a CSV data file

    :param filename: CSV filename containing a column header
    :param x_indices: integer or string indices
    :param target_vars: Target variables to use
    :param group: CSV column name containing grouping information
    :return: Tuple containing train data, target data, groups, features
    """
    import pandas as pd

    data = pd.read_csv(filename)
    if isinstance(x_indices[0], int):
        X = data.iloc[:, x_indices]
    elif isinstance(x_indices[0], str):
        X = data[x_indices]
    else:
        raise ValueError(f"{x_indices} is not a list of string or ints")
    Y = data[target_vars]
    if group is None:
        groups = list(range(X.shape[0]))
    else:
        groups = data[group].values
    feature_names = list(X.columns)
    return X.values, Y.values, groups, feature_names


def gen_splits(n_splits, test_size, X, Y, groups=None, random_state=0):
    """Generate train-test splits for the data.

    Uses GroupShuffleSplit from scikit-learn

    :param n_splits: Number of splits
    :param test_size: fractional test size
    :param X: Sample feature data
    :param Y: Sample target data
    :param groups: Grouping of sample data for shufflesplit
    :param random_state: randomization for shuffling (default 0)
    :return: splits and indices to splits
    """
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    train_test_splits = list(gss.split(X, Y, groups=groups))
    split_indices = list(range(n_splits))
    return train_test_splits, split_indices


def train_test_kernel(X, y, train_test_split, split_index, clf_info, permute):
    """Core model fitting and predicting function

    :param X: Input features
    :param y: Target variables
    :param train_test_split: split indices
    :param split_index: which index to use
    :param clf_info: how to construct the classifier
    :param permute: whether to run it in permuted mode or not
    :return: outputs, trained classifier with sample indices
    """
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


def calc_metric(output, metrics):
    """Calculate the scores for the predicted outputs

    :param output: true, predicted output
    :param metrics: list of metrics to evaluate
    :return: list of scores and pass the output
    """
    score = []
    for metric in metrics:
        metric_mod = __import__("sklearn.metrics", fromlist=[metric])
        metric_func = getattr(metric_mod, metric)
        score.append(metric_func(output[0], output[1]))
    return score, output


def get_shap(X, permute, model, gen_shap=False, nsamples="auto", l1_reg="aic"):
    """Compute shap information for the test data

    :param X: sample data
    :param permute: whether model was permuted or not
    :param model: model containing trained classifier and train/test index
    :param gen_shap: whether to generate shap features
    :param nsamples: number of samples for shap evaluation
    :param l1_reg: L1 regularization for shap evaluation
    :return: shap values for each test sample
    """
    if permute or not gen_shap:
        return []
    pipe, train_index, test_index = model
    import shap

    explainer = shap.KernelExplainer(pipe.predict, shap.kmeans(X[train_index], 5))
    shaps = explainer.shap_values(X[test_index], nsamples=nsamples, l1_reg=l1_reg)
    return shaps
