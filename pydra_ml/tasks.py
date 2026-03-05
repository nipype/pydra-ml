#!/usr/bin/env python

import typing as ty

from pydra.utils.hash import Cache, register_serializer
from sklearn.pipeline import Pipeline


@register_serializer
def bytes_repr_Pipeline(obj: Pipeline, cache: Cache):
    yield str(obj).encode()


try:
    from imblearn.pipeline import Pipeline as ImbPipeline

    @register_serializer
    def bytes_repr_ImbPipeline(obj: ImbPipeline, cache: Cache):
        yield str(obj).encode()

except ImportError:
    pass


def to_instance(clf_info):
    """Recursively instantiate a classifier/regressor from a spec list.

    :param clf_info: [module, class] or [module, class, {params}] or
        [module, class, {params}, grid] where params may contain nested
        clf_info specs under 'estimators' (list of [name, clf_info] pairs)
        and 'final_estimator' (a clf_info list) for meta-estimators like
        StackingRegressor / StackingClassifier.
    :return: instantiated sklearn-compatible estimator
    """
    mod = __import__(clf_info[0], fromlist=[clf_info[1]])
    params = {}
    if len(clf_info) > 2:
        params = dict(clf_info[2])
        if "estimators" in params:
            params["estimators"] = [
                (
                    (name, to_instance(est_info))
                    if isinstance(est_info, list)
                    else (name, est_info)
                )
                for name, est_info in params["estimators"]
            ]
        if "final_estimator" in params and isinstance(params["final_estimator"], list):
            params["final_estimator"] = to_instance(params["final_estimator"])
    clf = getattr(mod, clf_info[1])(**params)
    if len(clf_info) == 4:
        from sklearn.model_selection import GridSearchCV

        clf = GridSearchCV(clf, param_grid=clf_info[3])
    return clf


def _configure_group_cv(pipe, X_fit, y_fit, groups_fit):
    """Precompute group-aware CV splits and bake them into nested estimators.

    For each pipeline step that is a StackingRegressor, StackingClassifier,
    or GridSearchCV, replaces its ``cv`` with a precomputed list of
    (train, test) index tuples produced by ``GroupKFold``.  Baking the splits
    in avoids any need for sklearn metadata routing — the estimator uses the
    exact index lists and never needs ``groups`` passed at fit time.

    :param pipe: constructed sklearn / imblearn Pipeline
    :param X_fit: training feature matrix (used only for split generation)
    :param y_fit: training target array (used only for split generation)
    :param groups_fit: group labels for the training data (1-D array-like)
    :return: empty dict (no extra fit_params needed)
    """
    from sklearn.model_selection import GroupKFold

    _NESTED_CV = {"StackingRegressor", "StackingClassifier", "GridSearchCV"}

    for _step_name, step_est in pipe.steps:
        if type(step_est).__name__ in _NESTED_CV:
            n = step_est.cv if isinstance(step_est.cv, int) else 5
            gkf = GroupKFold(n_splits=n)
            step_est.cv = list(gkf.split(X_fit, y_fit, groups=groups_fit))

    return {}


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
    Y = data[list(target_vars)]
    if group is None:
        groups = list(range(X.shape[0]))
    else:
        groups = data[group].values
    feature_names = list(X.columns)
    return X.values, Y.values, groups, feature_names


def gen_splits(
    n_splits,
    test_size,
    X,
    Y,
    groups=None,
    random_state=0,
    bootstrap_strategy="group_shuffle",
    n_bins=10,
):
    """Generate train-test splits for the data.

    Uses GroupShuffleSplit (default), StratifiedShuffleSplit, or
    stratified-regression (quantile-binned StratifiedShuffleSplit).

    :param n_splits: Number of splits
    :param test_size: fractional test size
    :param X: Sample feature data
    :param Y: Sample target data
    :param groups: Grouping of sample data for shufflesplit (ignored for stratified strategies)
    :param random_state: randomization for shuffling (default 0)
    :param bootstrap_strategy: "group_shuffle" (default), "stratified", or
        "stratified_regression" (bins continuous Y into n_bins quantile bins then
        stratifies on those bins to ensure each split covers the full target range)
    :param n_bins: Number of quantile bins used by "stratified_regression" (default 10)
    :return: splits and indices to splits
    """
    if bootstrap_strategy == "stratified":
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        train_test_splits = list(splitter.split(X, Y.ravel()))
    elif bootstrap_strategy == "stratified_regression":
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.preprocessing import KBinsDiscretizer

        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        y_binned = kbd.fit_transform(Y.ravel().reshape(-1, 1)).ravel().astype(int)
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        train_test_splits = list(splitter.split(X, y_binned))
    else:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        train_test_splits = list(splitter.split(X, Y, groups=groups))
    split_indices = list(range(n_splits))
    return train_test_splits, split_indices


def _resample_for_regression(X_train, y_train, balancing, balancing_bins):
    """Resample regression training data by binning the continuous target.

    Bins y_train into quantile bins, applies an imblearn resampler using the
    binned labels, then recovers continuous y values via 1-NN lookup.  For
    under-sampled rows (exact copies of originals) k=1 NN returns the exact
    original y value.  For over-sampled synthetic rows the nearest original
    neighbour's y value is used as an approximation.

    :param X_train: training feature matrix
    :param y_train: continuous training target (1-D)
    :param balancing: imblearn resampler spec [module, class, {params}]
    :param balancing_bins: number of quantile bins for discretising y_train
    :return: (X_resampled, y_resampled)
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import KBinsDiscretizer

    kbd = KBinsDiscretizer(n_bins=balancing_bins, encode="ordinal", strategy="quantile")
    y_binned = kbd.fit_transform(y_train.reshape(-1, 1)).ravel().astype(int)

    mod = __import__(balancing[0], fromlist=[balancing[1]])
    params = balancing[2] if len(balancing) > 2 else {}
    resampler = getattr(mod, balancing[1])(**params)
    X_res, _ = resampler.fit_resample(X_train, y_binned)

    # Recover continuous y values.
    # k=1 NN on the original feature space: kept originals map to themselves
    # exactly; synthetic points get their nearest original neighbour's y value.
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_res = knn.predict(X_res)
    return X_res, y_res


def train_test_kernel(
    X,
    y,
    train_test_split,
    split_index,
    clf_info,
    permute,
    balancing=None,
    balancing_bins=None,
    groups=None,
):
    """Core model fitting and predicting function

    :param X: Input features
    :param y: Target variables
    :param train_test_split: split indices
    :param split_index: which index to use
    :param clf_info: how to construct the classifier
    :param permute: whether to run it in permuted mode or not
    :param balancing: optional imblearn resampler spec [module, class, {params}]
    :param balancing_bins: if set, enables regression-aware resampling — the
        resampler is applied pre-fit on binned y (not inside the pipeline) and
        continuous y values are recovered via 1-NN.  When None and balancing is
        set, the resampler is inserted into an imblearn Pipeline (classification).
    :param groups: optional group labels (1-D array-like, same length as X).
        When provided, nested estimators (StackingRegressor, GridSearchCV, …)
        inside the pipeline are automatically reconfigured to use GroupKFold and
        the training-subset groups are forwarded via pipe.fit() fit_params.
    :return: outputs, trained classifier with sample indices
    """
    import numpy as np
    from sklearn.pipeline import Pipeline

    def _is_imblearn(info):
        return info[0].startswith("imblearn")

    def _make_pipeline(steps, use_imblearn):
        if use_imblearn:
            from imblearn.pipeline import Pipeline as ImbPipeline

            return ImbPipeline(steps)
        return Pipeline(steps)

    train_index, test_index = train_test_split[split_index]
    y = y.ravel()
    if type(X[0][0]) is str:
        # it's loaded as bytes, so we need to decode as utf-8
        X = np.array([str.encode(n[0]).decode("utf-8") for n in X])

    if balancing is not None and balancing_bins is not None:
        # Regression resampling: resample training data before fitting.
        # The pipeline is built without a balancing step (resampling is pre-fit).
        X_fit, y_fit = _resample_for_regression(
            X[train_index], y[train_index], balancing, balancing_bins
        )
        if isinstance(clf_info[0], list):
            use_imblearn = any(_is_imblearn(val) for val in clf_info)
            steps = [(val[1], to_instance(val)) for val in clf_info]
            pipe = _make_pipeline(steps, use_imblearn)
        else:
            clf = to_instance(clf_info)
            from sklearn.preprocessing import StandardScaler

            pipe = Pipeline([("std", StandardScaler()), (clf_info[1], clf)])
    else:
        X_fit, y_fit = X[train_index], y[train_index]
        # Classification resampling: insert balancing step into imblearn Pipeline.
        if isinstance(clf_info[0], list):
            use_imblearn = any(_is_imblearn(val) for val in clf_info)
            steps = [(val[1], to_instance(val)) for val in clf_info]
            if balancing is not None:
                use_imblearn = True
                steps.insert(-1, (balancing[1], to_instance(balancing)))
            pipe = _make_pipeline(steps, use_imblearn)
        else:
            clf = to_instance(clf_info)
            from sklearn.preprocessing import StandardScaler

            steps = [("std", StandardScaler())]
            if balancing is not None:
                steps.append((balancing[1], to_instance(balancing)))
                use_imblearn = True
            else:
                use_imblearn = False
            steps.append((clf_info[1], clf))
            pipe = _make_pipeline(steps, use_imblearn)

    if groups is not None:
        groups_fit = np.asarray(groups)[train_index]
        _configure_group_cv(pipe, X_fit, y_fit, groups_fit)

    if permute:
        if groups is not None:
            # Block permutation: shuffle group→y mapping across participants
            # (keep within-participant y values together, reassign them to a
            # different participant's features).
            unique_grps = np.unique(groups_fit)
            shuffled_grps = unique_grps[np.random.permutation(len(unique_grps))]
            y_permuted = y_fit.copy()
            for orig_grp, src_grp in zip(unique_grps, shuffled_grps):
                orig_idx = np.where(groups_fit == orig_grp)[0]
                src_idx = np.where(groups_fit == src_grp)[0]
                src_y = y_fit[src_idx]
                for k, i in enumerate(orig_idx):
                    y_permuted[i] = src_y[k % len(src_y)]
            pipe.fit(X_fit, y_permuted)
        else:
            pipe.fit(X_fit, y_fit[np.random.permutation(range(len(y_fit)))])
    else:
        pipe.fit(X_fit, y_fit)
    predicted = pipe.predict(X[test_index])
    try:
        predicted_proba = pipe.predict_proba(X[test_index])
    except AttributeError:
        predicted_proba = None
    return (y[test_index], predicted, predicted_proba), (pipe, train_index, test_index)


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
        if metric == "roc_auc_score" and output[2] is not None:
            # For roc_auc_score, we need to pass the probability of the positive class
            score.append(metric_func(output[0], output[2][:, 1]))
        else:
            score.append(metric_func(output[0], output[1]))
    return score, output


def get_feature_importance(
    *,
    permute: bool,
    model: ty.Tuple[Pipeline, list, list],
    gen_feature_importance: bool = True,
):
    """Compute feature importance for the model

    Parameters
    ----------
    permute : bool
        Whether or not to run the model in permuted mode
    model : tuple(sklearn.pipeline.Pipeline, list, list)
        The model to compute feature importance for
    gen_feature_importance : bool
        Whether or not to generate the feature importance
    Returns
    -------
    list
        List of feature importance
    """
    if permute or not gen_feature_importance:
        return []
    pipeline, train_index, test_index = model
    pipeline_steps = pipeline.steps[-1][1]
    model_name = str(pipeline_steps)
    # Each model type may have a different method or none at all.
    # See here for sklearn models: https://scikit-learn.org/stable/supervised_learning.html
    tree_models = [
        "Tree",
        "Forest",
        "Boost",
        "XGB",
    ]  # not available for Bagging methods, voting methods or  'xgboost' library models.
    if any(n in model_name for n in tree_models):
        # Tree model is in model_name
        feature_importance = (
            pipeline_steps.feature_importances_
        )  # for decision tree, Random Forest, or boosting algorithms
    elif "MLP" in model_name:
        feature_importance = (
            pipeline_steps.coefs_
        )  # for multi-layer perceptron, which returns a list
    # elif 'LinearRegression' in model_name:
    # 	feature_importance = pipeline.coef_  # for LinearRegression in particular
    else:
        try:
            feature_importance = pipeline_steps.coef_  # for linear models
        except AttributeError as e:
            import warnings

            warnings.warn(
                f""""

                Warning: you set gen_feature_importance to true, but it
                could not be computed and will be returned as an empty list
                because after running this

                pipeline_steps = pipeline.steps[-1][1]

                none of the following methods worked:

                pipeline_steps.feature_importances_
                pipeline_steps.coefs_
                pipeline_steps.coef_

                Please add correct method in tasks.py or if non-existent,
                set gen_feature_importance to false in the spec file.

                This is the error that was returned by sklearn:\n\t{e}\n
                """
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
        pipe.steps[-1][1],
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
    shaps = explainer.shap_values(
        X[test_index], nsamples=nsamples, l1_reg=l1_reg, silent=True
    )
    return shaps


def create_model(
    X, y, clf_info, permute, balancing=None, balancing_bins=None, groups=None
):
    """Train a model with all the data

    :param X: Input features
    :param y: Target variables
    :param clf_info: how to construct the classifier
    :param permute: whether to run it in permuted mode or not
    :param balancing: optional imblearn resampler spec [module, class, {params}]
    :param balancing_bins: if set, enables regression-aware resampling (see
        train_test_kernel for details)
    :param groups: optional group labels (1-D array-like). When provided,
        nested estimators are reconfigured to use GroupKFold and groups are
        forwarded via pipe.fit() fit_params.
    :return: training error, classifier
    """
    import numpy as np
    from sklearn.pipeline import Pipeline

    def _is_imblearn(info):
        return info[0].startswith("imblearn")

    def _make_pipeline(steps, use_imblearn):
        if use_imblearn:
            from imblearn.pipeline import Pipeline as ImbPipeline

            return ImbPipeline(steps)
        return Pipeline(steps)

    y = y.ravel()

    if balancing is not None and balancing_bins is not None:
        # Regression resampling: resample all data before fitting.
        X_fit, y_fit = _resample_for_regression(X, y, balancing, balancing_bins)
        if isinstance(clf_info[0], list):
            use_imblearn = any(_is_imblearn(val) for val in clf_info)
            steps = [(val[1], to_instance(val)) for val in clf_info]
            pipe = _make_pipeline(steps, use_imblearn)
        else:
            clf = to_instance(clf_info)
            from sklearn.preprocessing import StandardScaler

            pipe = Pipeline([("std", StandardScaler()), (clf_info[1], clf)])
    else:
        X_fit, y_fit = X, y
        if isinstance(clf_info[0], list):
            use_imblearn = any(_is_imblearn(val) for val in clf_info)
            steps = [(val[1], to_instance(val)) for val in clf_info]
            if balancing is not None:
                use_imblearn = True
                steps.insert(-1, (balancing[1], to_instance(balancing)))
            pipe = _make_pipeline(steps, use_imblearn)
        else:
            clf = to_instance(clf_info)
            from sklearn.preprocessing import StandardScaler

            steps = [("std", StandardScaler())]
            if balancing is not None:
                steps.append((balancing[1], to_instance(balancing)))
                use_imblearn = True
            else:
                use_imblearn = False
            steps.append((clf_info[1], clf))
            pipe = _make_pipeline(steps, use_imblearn)

    if groups is not None:
        import numpy as _np

        groups_arr = _np.asarray(groups)
        _configure_group_cv(pipe, X_fit, y_fit, groups_arr)

    if permute:
        if groups is not None:
            # Block permutation: shuffle group→y mapping across participants
            # (keep within-participant y values together, reassign them to a
            # different participant's features).
            unique_grps = _np.unique(groups_arr)
            shuffled_grps = unique_grps[_np.random.permutation(len(unique_grps))]
            y_permuted = y_fit.copy()
            for orig_grp, src_grp in zip(unique_grps, shuffled_grps):
                orig_idx = _np.where(groups_arr == orig_grp)[0]
                src_idx = _np.where(groups_arr == src_grp)[0]
                src_y = y_fit[src_idx]
                for k, i in enumerate(orig_idx):
                    y_permuted[i] = src_y[k % len(src_y)]
            pipe.fit(X_fit, y_permuted)
        else:
            pipe.fit(X_fit, y_fit[np.random.permutation(range(len(y_fit)))])
    else:
        pipe.fit(X_fit, y_fit)
    predicted = pipe.predict(X)
    return (y, predicted), pipe
