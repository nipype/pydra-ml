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
    from sklearn.pipeline import Pipeline
    import numpy as np

    def to_instance(clf_info):
        mod = __import__(clf_info[0], fromlist=[clf_info[1]])
        params = {}
        if len(clf_info) > 2:
            params = clf_info[2]
        clf = getattr(mod, clf_info[1])(**params)
        if len(clf_info) == 4:
            from sklearn.model_selection import GridSearchCV

            clf = GridSearchCV(clf, param_grid=clf_info[3])
        return clf

    if isinstance(clf_info[0], list):
        # Process as a pipeline constructor
        steps = []
        for val in clf_info:
            step = to_instance(val)
            steps.append((val[1], step))
        pipe = Pipeline(steps)
    else:
        clf = to_instance(clf_info)
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("std", StandardScaler()), (clf_info[1], clf)])

    train_index, test_index = train_test_split[split_index]
    y = y.ravel()
    if type(X[0][0]) == str:
        # it's loaded as bytes, so we need to decode as utf-8
        X = np.array([str.encode(n[0]).decode("utf-8") for n in X])
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


def get_feature_importance(permute, model, gen_feature_importance=True):
    if permute or not gen_feature_importance:
        return []
    pipe, train_index, test_index = model
    pipeline = pipe.steps[1][1]
    model_name = str(pipe.steps[1][1])
    # Each model type may have a different method or none at all. See here for sklearn models: https://scikit-learn.org/stable/supervised_learning.html
    tree_models = [
        "Tree",
        "Forest",
        "Boost",
        "XGB",
    ]  # not available for Bagging methods, voting methods or  'xgboost' library models.
    if any(n in model_name for n in tree_models):
        # Tree model is in model_name
        feature_importance = (
            pipeline.feature_importances_
        )  # for decision tree, Random Forest, or boosting algorithms
    elif "MLP" in model_name:
        feature_importance = (
            pipeline.coefs_
        )  # for multi-layer perceptron, which returns a list
    # elif 'LinearRegression' in model_name:
    # 	feature_importance = pipeline.coef_  # for LinearRegression in particular
    else:
        try:
            feature_importance = pipeline.coef_  # for linear models
        except:
            import warnings

            warnings.warn(
                f"Warning: gen_feature_importance() could not be computed because the following methods did not work: pipeline.feature_importances_, pipeline.coefs_, pipeline.coef_ . Please add correct method in tasks.py or if inexistent, set gen_feature_importance to false."
            )
            feature_importance = []
    return feature_importance


def get_permutation_importance(
    X,
    y,
    permute,
    model,
    permutation_importance_n_repeats=5,
    permutation_importance_scoring=None,
    gen_permutation_importance=True,
):
    if permute or not gen_permutation_importance:
        return []
    from sklearn.inspection import permutation_importance

    pipe, train_index, test_index = model
    results = permutation_importance(
        pipe.steps[1][1],
        X[test_index],
        y[test_index],
        scoring=permutation_importance_scoring,
        n_repeats=permutation_importance_n_repeats,
    )
    permutation_feature_importance = results.importances_mean
    return permutation_feature_importance


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


def create_model(X, y, clf_info, permute):
    """Train a model with all the data

    :param X: Input features
    :param y: Target variables
    :param clf_info: how to construct the classifier
    :param permute: whether to run it in permuted mode or not
    :return: training error, classifier
    """
    from sklearn.pipeline import Pipeline
    import numpy as np

    def to_instance(clf_info):
        mod = __import__(clf_info[0], fromlist=[clf_info[1]])
        params = {}
        if len(clf_info) > 2:
            params = clf_info[2]
        clf = getattr(mod, clf_info[1])(**params)
        if len(clf_info) == 4:
            from sklearn.model_selection import GridSearchCV

            clf = GridSearchCV(clf, param_grid=clf_info[3])
        return clf

    if isinstance(clf_info[0], list):
        # Process as a pipeline constructor
        steps = []
        for val in clf_info:
            step = to_instance(val)
            steps.append((val[1], step))
        pipe = Pipeline(steps)
    else:
        clf = to_instance(clf_info)
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("std", StandardScaler()), (clf_info[1], clf)])

    y = y.ravel()
    if permute:
        pipe.fit(X, y[np.random.permutation(range(len(y)))])
    else:
        pipe.fit(X, y)
    predicted = pipe.predict(X)
    return (y, predicted), pipe
