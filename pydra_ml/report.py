import datetime
import hashlib
import os
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, explained_variance_score

matplotlib.use("Agg")


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def performance_table(df, output_dir, round_decimals=2):
    timestamp = datetime.datetime.utcnow().isoformat()
    timestamp = timestamp.replace(":", "").replace("-", "")
    output_dir = output_dir + f"performance_table-{timestamp}/"
    os.mkdir(output_dir)

    classifier_names = sorted(df.Classifier.unique())

    for idx, metric in enumerate(df.metric.unique()):
        df_metric = df[df.metric == metric]
        # data
        df_metric_data = df_metric[df_metric.type == "data"]
        df_metric_data = df_metric_data[["score", "Classifier"]]
        dfp = df_metric_data.pivot(columns="Classifier")
        dfp.columns = dfp.columns.droplevel()
        df_metric_data_clean = pd.DataFrame()
        for clf in classifier_names:
            df_clf = dfp[clf].dropna()
            df_clf = df_clf.reset_index(drop=True)
            df_metric_data_clean[clf] = df_clf
        df_metric_data_clean_median = df_metric_data_clean.median().T
        df_metric_data_clean.loc[len(df_metric_data_clean)] = (
            df_metric_data_clean_median
        )
        df_metric_data_clean.index = list(df_metric_data_clean.index[:-1]) + ["median"]
        df_metric_data_clean.to_csv(
            os.path.join(
                output_dir,
                f"test-performance-table_{metric}_all-splits_{timestamp}.csv",
            ),
            float_format="%.2f",
        )

        # null
        if "null" in df_metric.type.unique():
            df_metric_null = df_metric[df_metric.type == "null"]
            df_metric_null = df_metric_null[["score", "Classifier"]]
            dfp = df_metric_null.pivot(columns="Classifier")
            dfp.columns = dfp.columns.droplevel()
            df_metric_null_clean = pd.DataFrame()
            for clf in df_metric_data.Classifier.unique():
                df_clf = dfp[clf].dropna()
                df_clf = df_clf.reset_index(drop=True)
                df_metric_null_clean[clf] = df_clf
            df_metric_null_clean_median = df_metric_null_clean.median().T.round(2)

        # Save median score with median null score in square brackets as strings

        df_summary = pd.DataFrame(index=[0], columns=classifier_names)
        df_summary[classifier_names] = ""

        for clf in classifier_names:
            data_median = round(df_metric_data_clean_median[clf], round_decimals)
            # 2.5-97.5 percentile interval of the per-split scores. This is
            # NOT a true confidence interval -- the splits are overlapping,
            # non-independent bootstrap resamples, so there is no coverage
            # guarantee -- it is only descriptive of the observed spread.
            interval_lower = round(
                np.percentile(df_metric_data_clean[clf].values, 2.5), round_decimals
            )
            interval_upper = round(
                np.percentile(df_metric_data_clean[clf].values, 97.5), round_decimals
            )
            if "null" in df_metric.type.unique():
                null_median = round(df_metric_null_clean_median[clf], 2)
                df_summary.loc[0, clf] = (
                    f"{data_median} [{interval_lower}–{interval_upper}; {null_median}]"
                )
            else:
                df_summary.loc[0, clf] = (
                    f"{data_median} [{interval_lower}–{interval_upper}]"
                )

        df_summary.to_csv(
            os.path.join(
                output_dir,
                f"test-performance-table_{metric}_with-95interval_{timestamp}.csv",
            )
        )
    return


def plot_summary(summary, output_dir=None, filename="shap_plot", plot_top_n_shap=16):
    plt.clf()
    plt.figure(figsize=(8, 12))
    # plot without all bootstrapping values
    summary = summary[["mean", "std", "min", "max"]]
    num_features = len(list(summary.index))
    if (plot_top_n_shap != 1 and type(plot_top_n_shap) is float) or type(
        plot_top_n_shap
    ) is int:
        # if plot_top_n_shap != 1.0 but includes 1 (int)
        if plot_top_n_shap <= 0:
            raise ValueError(
                """plot_top_n_shap should be a float between 0 and 1.0 or an
                integer >= 1. You set to zero or negative."""
            )
        elif plot_top_n_shap < 1:
            plot_top_n_shap = int(np.round(plot_top_n_shap * num_features))
        summary = summary.iloc[:plot_top_n_shap, :]
        filename += f"_top_{plot_top_n_shap}"

    hm = sns.heatmap(
        summary.round(3),
        annot=True,
        xticklabels=True,
        yticklabels=True,
        cbar=False,
        square=True,
        annot_kws={"size": 12},
    )
    hm.set_xticklabels(summary.columns, rotation=45)
    hm.set_yticklabels(summary.index, rotation=0)
    plt.ylabel("Features")
    plt.show(block=False)
    plt.savefig(output_dir + f"summary_{filename}.png", dpi=100, bbox_inches="tight")


def shaps_to_summary(
    shaps_n_splits,
    feature_names=None,
    output_dir=None,
    filename="shap_summary",
    plot_top_n_shap=16,
):
    shaps_n_splits.columns = [
        "split_{}".format(n) for n in range(shaps_n_splits.shape[1])
    ]
    if feature_names:
        shaps_n_splits.index = feature_names
    # else:
    # 	shaps_n_splits.index = [str(n) for n in shaps_n_splits.index]
    # add summary stats, computed from the split columns only (not from
    # each other, e.g. "max" must not include the "mean"/"std" columns)
    split_cols = shaps_n_splits.columns
    shaps_n_splits["mean"] = shaps_n_splits[split_cols].mean(axis=1)
    shaps_n_splits["std"] = shaps_n_splits[split_cols].std(axis=1)
    shaps_n_splits["min"] = shaps_n_splits[split_cols].min(axis=1)
    shaps_n_splits["max"] = shaps_n_splits[split_cols].max(axis=1)
    shaps_n_splits_sorted = shaps_n_splits.sort_values("mean")[::-1]
    shaps_n_splits_sorted.to_csv(f"{output_dir}summary_values_{filename}.csv")

    plot_summary(
        shaps_n_splits_sorted,
        output_dir=output_dir,
        filename=filename,
        plot_top_n_shap=plot_top_n_shap,
    )


def gen_report_shap_class(results, output_dir="./", plot_top_n_shap=16):
    # Create shap_dir
    timestamp = datetime.datetime.utcnow().isoformat()
    timestamp = timestamp.replace(":", "").replace("-", "")
    shap_dir = output_dir + f"shap-{timestamp}/"
    os.mkdir(shap_dir)

    feature_names = results[0][1].output.feature_names
    # save all TP, TN, FP, FN indexes
    indexes_all = {}
    labels = _clf_labels([r[0].get("ml_wf.clf_info") for r in results])

    for model_results in results:
        model_name = labels[repr(model_results[0].get("ml_wf.clf_info"))]
        indexes_all[model_name] = []
        shaps = model_results[1].output.shaps
        # this is (N, P, F) N splits, P predictions, F feature_names
        # make sure there are shap values (the
        if np.array(shaps[0]).size == 0:
            continue

        y_true_and_preds = model_results[1].output.output
        n_splits = len(y_true_and_preds)

        shaps_n_splits = {
            "all": [],
            "tp": [],
            "tn": [],
            "fp": [],
            "fn": [],
        }
        # this is key with shape (F, N) where F is feature_names, N is mean
        # shap values across splits

        # Obtain values for each bootstrapping split, then append summary
        # statistics to shaps_n_splits
        for split_i in range(n_splits):
            shaps_i = shaps[split_i]  # all shap values for this bootstrapping split
            y_true = y_true_and_preds[split_i][0]
            y_pred = y_true_and_preds[split_i][1]

            split_performance = accuracy_score(y_true, y_pred)

            # split prediction indexes into TP, TN, FP, FN, good for error auditing
            indexes = {"tp": [], "tn": [], "fp": [], "fn": []}
            for i in range(len(y_true)):
                if y_true[i] == y_pred[i] and y_pred[i] == 1:
                    indexes["tp"].append(i)
                elif y_true[i] == y_pred[i] and y_pred[i] == 0:
                    indexes["tn"].append(i)
                elif y_true[i] != y_pred[i] and y_pred[i] == 1:
                    indexes["fp"].append(i)
                elif y_true[i] != y_pred[i] and y_pred[i] == 0:
                    indexes["fn"].append(i)
            indexes_all[model_name].append(indexes)

            #  For each quadrant, obtain F shap values for P predictions,
            # take the absolute mean weighted by performance across all predictions
            for quadrant in ["tp", "tn", "fp", "fn"]:
                if len(indexes.get(quadrant)) == 0:
                    warnings.warn(
                        f"""There were no {quadrant.upper()}s, this will output NaNs
                        in the csv and figure for this split column"""
                    )
                shaps_i_quadrant = np.array(shaps_i)[
                    indexes.get(quadrant)
                ]  # shape (P, F) P prediction x F feature_names
                abs_weighted_shap_values = np.abs(shaps_i_quadrant) * split_performance
                shaps_n_splits[quadrant].append(
                    np.mean(abs_weighted_shap_values, axis=0)
                )
            #  obtain F shap values for P predictions, take the absolute mean
            #  weighted by performance across all predictions
            abs_weighted_shap_values = np.abs(shaps_i) * split_performance
            shaps_n_splits["all"].append(np.mean(abs_weighted_shap_values, axis=0))

        # Build df for summary statistics for each quadrant
        for quadrant in ["tp", "tn", "fp", "fn"]:
            shaps_n_splits_quadrant = pd.DataFrame(shaps_n_splits.get(quadrant)).T
            shaps_to_summary(
                shaps_n_splits_quadrant,
                feature_names,
                output_dir=shap_dir,
                filename=f"shap_{model_name}_{quadrant}",
                plot_top_n_shap=plot_top_n_shap,
            )

        # Single csv for all predictions
        shaps_n_splits_all = pd.DataFrame(shaps_n_splits.get("all")).T
        shaps_to_summary(
            shaps_n_splits_all,
            feature_names,
            output_dir=shap_dir,
            filename=f"shap_{model_name}_all_predictions",
            plot_top_n_shap=plot_top_n_shap,
        )
    save_obj(indexes_all, shap_dir + "indexes_quadrant.pkl")


def gen_report_shap_regres(results, output_dir="./", plot_top_n_shap=16):
    # Create shap_dir
    timestamp = datetime.datetime.utcnow().isoformat()
    timestamp = timestamp.replace(":", "").replace("-", "")
    shap_dir = output_dir + f"shap-{timestamp}/"
    os.mkdir(shap_dir)

    feature_names = results[0][1].output.feature_names
    # save all TP, TN, FP, FN indexes
    indexes_all = {}
    labels = _clf_labels([r[0].get("ml_wf.clf_info") for r in results])

    for model_results in results:
        model_name = labels[repr(model_results[0].get("ml_wf.clf_info"))]
        indexes_all[model_name] = []
        shaps = model_results[1].output.shaps
        # this is (N, P, F) N splits, P predictions, F feature_names
        # make sure there are shap values (the
        if np.array(shaps[0]).size == 0:
            continue

        y_true_and_preds = model_results[1].output.output
        n_splits = len(y_true_and_preds)

        shaps_n_splits = {
            "all": [],
            "lp": [],
            "lm": [],
            "um": [],
            "up": [],
        }
        #   this is key with shape (F, N) where F is feature_names,
        #   N is mean shap values across splits

        # Obtain values for each bootstrapping split,
        # then append summary statistics to shaps_n_splits
        for split_i in range(n_splits):
            shaps_i = shaps[split_i]  # all shap values for this bootstrapping split
            y_true = y_true_and_preds[split_i][0]
            y_pred = y_true_and_preds[split_i][1]
            # Clip to [0, 1]: explained_variance_score is unbounded below for
            # a poor model, and an unbounded negative weight would flip the
            # sign of (and so invert the ranking of) the weighted SHAP values.
            split_performance = max(explained_variance_score(y_true, y_pred), 0.0)

            # split prediction indexes into upper, median, lower, good for error auditing
            indexes = {"lp": [], "lm": [], "um": [], "up": []}
            q = np.array([25, 50, 75])
            prc = np.percentile(y_true, q)
            for i in range(len(y_true)):
                if prc[0] >= y_pred[i]:
                    indexes["lp"].append(i)
                elif prc[0] < y_pred[i] and prc[1] >= y_pred[i]:
                    indexes["lm"].append(i)
                elif prc[1] < y_pred[i] and prc[2] >= y_pred[i]:
                    indexes["um"].append(i)
                elif prc[2] < y_pred[i]:
                    indexes["up"].append(i)
            indexes_all[model_name].append(indexes)

            #  For each quadrant, obtain F shap values for P predictions,
            #  take the absolute mean weighted by performance across all predictions
            for quadrant in ["lp", "lm", "um", "up"]:
                if len(indexes.get(quadrant)) == 0:
                    warnings.warn(
                        f"""There were no {quadrant.upper()}s, this will
                        output NaNs in the csv and figure for this split column"""
                    )
                shaps_i_quadrant = np.array(shaps_i)[
                    indexes.get(quadrant)
                ]  # shape (P, F) P prediction x F feature_names
                abs_weighted_shap_values = np.abs(shaps_i_quadrant) * split_performance
                shaps_n_splits[quadrant].append(
                    np.mean(abs_weighted_shap_values, axis=0)
                )
            #  obtain F shap values for P predictions, take the absolute mean weighted
            #  by performance across all predictions
            abs_weighted_shap_values = np.abs(shaps_i) * split_performance
            shaps_n_splits["all"].append(np.mean(abs_weighted_shap_values, axis=0))

        # Build df for summary statistics for each quadrant
        for quadrant in ["lp", "lm", "um", "up"]:
            shaps_n_splits_quadrant = pd.DataFrame(shaps_n_splits.get(quadrant)).T
            shaps_to_summary(
                shaps_n_splits_quadrant,
                feature_names,
                output_dir=shap_dir,
                filename=f"shap_{model_name}_{quadrant}",
                plot_top_n_shap=plot_top_n_shap,
            )

        # Single csv for all predictions
        shaps_n_splits_all = pd.DataFrame(shaps_n_splits.get("all")).T
        shaps_to_summary(
            shaps_n_splits_all,
            feature_names,
            output_dir=shap_dir,
            filename=f"shap_{model_name}_all_predictions",
            plot_top_n_shap=plot_top_n_shap,
        )
    save_obj(indexes_all, shap_dir + "indexes_quadrant.pkl")


def permutation_test_pvalue(mean_score, distribution):
    """
    the permutation-based empirical p-value from Test 1 in:
    Ojala and Garriga. Permutation Tests for Studying
    Classifier Performance. The Journal of Machine Learning
    Research (2010) vol. 11
    Based off of:
    github.com/scikit-learn/scikit-learn/blob
    /15a949460dbf19e5e196b8ef48f9712b72a3b3c3
    /sklearn/model_selection/_validation.py#L1062
    """

    n_distribution = len(distribution)
    pvalue = (np.sum(np.array(distribution) >= mean_score) + 1.0) / (n_distribution + 1)
    return pvalue


_LOWER_IS_BETTER_METRICS = {"brier_score_loss"}


def corrected_resampled_ttest(diffs, test_train_ratio):
    """Nadeau & Bengio (2003) corrected resampled paired t-test.

    A plain paired t-test on per-split score differences assumes the splits
    are independent. They are not: repeated random subsampling (as used by
    GroupShuffleSplit here) draws overlapping train/test sets across splits,
    which correlates the differences and makes a naive t-test underestimate
    variance (inflating false positives) -- the same fold/split-dependence
    problem described in Nadeau & Bengio (2003), "Inference for the
    Generalization Error". Their correction inflates the naive variance by
    (1/n + n_test/n_train), using the realized test/train sample-count ratio
    in place of the (unobservable, without something like repeated
    split-halves) true between-split correlation.

    :param diffs: Per-split (model A - model B) score differences, all from
        the same n_splits GroupShuffleSplit partition shared across models.
    :param test_train_ratio: The realized mean test/train sample-count ratio
        across those splits (see `tasks.gen_splits`) -- not the nominal
        `test_size` fraction, since GroupShuffleSplit selects whole groups
        and the realized sample-level ratio can differ from the nominal
        fraction when group sizes are uneven, and this also stays correct
        if a k-fold-style splitter is used instead (where it would work out
        to roughly 1/(k-1)).
    :return: (t_stat, p_value), two-sided, df = n - 1. Both are NaN if fewer
        than 2 finite differences are available.
    """
    if not (isinstance(test_train_ratio, (int, float)) and test_train_ratio > 0):
        raise ValueError(
            "test_train_ratio must be a positive number, got " f"{test_train_ratio!r}"
        )
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n < 2:
        return np.nan, np.nan
    mean_diff = diffs.mean()
    var_diff = diffs.var(ddof=1)
    correction = 1.0 / n + test_train_ratio
    corrected_var = correction * var_diff
    if corrected_var == 0:
        if mean_diff == 0:
            return 0.0, 1.0
        # Every per-split difference is identical: no finite sample can
        # prove certainty, so fall back to the exact two-sided sign-test
        # p-value (probability all n diffs share a sign under the null)
        # rather than asserting an impossible p=0.
        return np.sign(mean_diff) * np.inf, 2.0 ** -(n - 1)
    t_stat = mean_diff / np.sqrt(corrected_var)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 1)
    return t_stat, p_value


def _holm_adjust(pvalues):
    """Holm-Bonferroni step-down adjustment for a flat list of p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = pvalues[idx]
        adjusted_p = (m - rank) * p if np.isfinite(p) else p
        running_max = (
            max(running_max, adjusted_p) if np.isfinite(adjusted_p) else running_max
        )
        adjusted[idx] = min(running_max, 1.0) if np.isfinite(adjusted_p) else adjusted_p
    return adjusted


def compute_pairwise_stats(df, test_train_ratio, metric_name=None):
    """Compare each pair of classifiers' per-split scores.

    When comparing a classifier to itself, compare to its own null
    (permuted-label) distribution with the Ojala & Garriga (2010) Test 1
    permutation p-value. Unlike the off-diagonal comparison below, this is
    conservative rather than exactly calibrated: it compares a *mean* over
    splits to a reference distribution of *individual* per-split null
    scores, whose spread is larger than that of the mean, so it under- (not
    over-) rejects. It also has a coarse floor of 1/(n_splits+1) on the
    achievable p-value.

    Comparing two different classifiers instead uses a paired test on their
    per-split score differences (joined on the "split" column, so the two
    classifiers are compared on the exact same train/test partitions), via
    `corrected_resampled_ttest`, which -- unlike a naive paired t-test or
    comparing one classifier's mean to another's raw score distribution --
    accounts for the correlation those shared, overlapping splits induce.

    :param df: Columns Classifier, type, score, split, with type referring
        to either the data distribution or the null distribution and split
        identifying the shared train/test partition. Each classifier must
        contribute at most one "data" row per split -- e.g. via
        `report._clf_labels`, which guarantees distinct classifiers get
        distinct labels -- or the paired comparison silently turns into a
        many-to-many join.
    :param test_train_ratio: Passed through to `corrected_resampled_ttest`.
    :param metric_name: The metric these scores are for. Used only to flip
        comparison direction for metrics in `_LOWER_IS_BETTER_METRICS`
        (e.g. brier_score_loss), where a lower score is the better one.
    :return: (effects, pvalues, adjusted_pvalues) -- Holm-corrected p-values
        for the off-diagonal (between-classifier) entries only, since the
        diagonal (self-vs-null) entries are a different test family; the
        diagonal is passed through unadjusted.
    """
    sign = -1.0 if metric_name in _LOWER_IS_BETTER_METRICS else 1.0
    order = [group[0] for group in df.groupby("Classifier")]
    N = len(order)
    effects = np.zeros((N, N)) * np.nan
    pvalues = np.zeros((N, N)) * np.nan
    data_by_clf = {}
    null_by_clf = {}
    for name, group in df.groupby("Classifier"):
        group_data = group[group["type"] == "data"]
        counts = group_data.groupby("split").size()
        if (counts > 1).any():
            raise ValueError(
                f"Classifier {name!r} has more than one 'data' row for the "
                "same split -- classifier display names must be unique "
                "(see report._clf_labels), otherwise the paired comparison "
                "silently turns into a many-to-many join."
            )
        data_by_clf[name] = group_data
        null_by_clf[name] = group[group["type"] == "null"]

    off_diag_positions = []
    off_diag_pvalues = []
    for idx1, name1 in enumerate(order):
        group1df = data_by_clf[name1]
        group1nulldf = null_by_clf[name1]
        for idx2, name2 in enumerate(order):
            group2df = data_by_clf[name2]
            if name1 != name2:
                paired = group1df.merge(group2df, on="split", suffixes=("_1", "_2"))
                diffs = sign * (paired["score_1"].values - paired["score_2"].values)
                stat, pval = corrected_resampled_ttest(diffs, test_train_ratio)
                off_diag_positions.append((idx1, idx2))
                off_diag_pvalues.append(pval)
            else:
                mean_score = sign * np.mean(group1df["score"].values)
                distribution = sign * group1nulldf["score"].values
                pval = permutation_test_pvalue(mean_score, distribution)
                stat = 0  # no comparable t-statistic for the null-distribution test

            effects[idx1, idx2] = stat
            pvalues[idx1, idx2] = pval

    adjusted_pvalues = pvalues.copy()
    if off_diag_pvalues:
        holm = _holm_adjust(off_diag_pvalues)
        for (idx1, idx2), adj in zip(off_diag_positions, holm):
            adjusted_pvalues[idx1, idx2] = adj
    return effects, pvalues, adjusted_pvalues


def _clf_base_name(clf_info):
    """Derive the short display name of a clf_info entry from its final step."""
    if isinstance(clf_info[0], list):
        base = clf_info[-1][1]
    else:
        base = clf_info[1]
    if "Classifier" in base:
        name = base.split("Classifier")[0]
    else:
        name = base.split("Regressor")[0]
    return name.split("CV")[0]


def _clf_labels(clf_infos):
    """Map each distinct clf_info (by repr) in a report to a display label.

    Two different clf_info entries -- e.g. the same estimator with
    different hyperparameters, or different pipelines that happen to end
    in the same final estimator -- reduce to the same base name from
    `_clf_base_name` alone, which silently merges their results into one
    row/column of the report. Disambiguate any base name shared by more
    than one distinct clf_info with a short deterministic hash of the
    full clf_info.
    """
    by_base = {}
    for clf_info in clf_infos:
        key = repr(clf_info)
        base = _clf_base_name(clf_info)
        by_base.setdefault(base, {})[key] = clf_info

    labels = {}
    for base, variants in by_base.items():
        if len(variants) == 1:
            (key,) = variants.keys()
            labels[key] = base
        else:
            for key in variants:
                digest = hashlib.sha256(key.encode()).hexdigest()[:6]
                labels[key] = f"{base}-{digest}"

    seen = {}
    for key, label in labels.items():
        if seen.get(label, key) != key:
            raise ValueError(
                f"clf_info entries {seen[label]!r} and {key!r} both produced "
                f"the report label {label!r}; please report this as a bug."
            )
        seen[label] = key
    return labels


def gen_report(
    results,
    prefix,
    metrics,
    *,
    test_train_ratio,
    gen_shap=True,
    output_dir="./",
    plot_top_n_shap=16,
):
    if len(results) == 0:
        raise ValueError("results is empty")
    labels = _clf_labels([val[0][prefix + ".clf_info"] for val in results])
    df = None
    for val in results:
        score = val[1].output.score
        if not isinstance(score, list):
            score = [score]

        clf = val[0][prefix + ".clf_info"]
        name = labels[repr(clf)]
        permute = val[0][prefix + ".permute"]
        for split_idx, scoreval in enumerate(score):
            for idx, metric in enumerate(metrics):
                new_row = {
                    "Classifier": name,
                    "type": "null" if permute else "data",
                    "metric": metrics[idx],
                    "score": scoreval[idx] if scoreval[idx] is not None else np.nan,
                    "split": split_idx,
                }
                if df is None:
                    df = pd.DataFrame([new_row])
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Generate table of median performance with 95% interval across splits
    # df_all = performance_table(results, prefix, output_dir, metrics, round_decimals=2)
    performance_table(df, output_dir, round_decimals=2)

    # Plot distribution of scores
    import datetime

    order = [group[0] for group in df.groupby("Classifier")]
    for name, subdf in df.groupby("metric"):
        sns.set(style="whitegrid", palette="pastel", color_codes=True)
        sns.set_context("talk")
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(
            x="Classifier",
            y="score",
            hue="type",
            data=subdf,
            split=True,
            inner="quartile",
            hue_order=["data", "null"],
            order=order,
        )
        ax.xaxis.set_ticks_position("top")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
        ax.set_ylabel(name)
        ax.legend(loc="center right", bbox_to_anchor=(1.2, 0.5), ncol=1)
        ax.tick_params(axis="both", which="both", length=0)
        sns.despine(left=True)
        plt.tight_layout()

        timestamp = datetime.datetime.utcnow().isoformat()
        timestamp = timestamp.replace(":", "").replace("-", "")
        plt.savefig(f"test-{name}-{timestamp}.png")
        plt.close()

        # Create comparison stats table if the metric is a score
        if "score" in name:
            effects, pvalues, adjusted_pvalues = compute_pairwise_stats(
                subdf, test_train_ratio, metric_name=name
            )
            sns.set(style="whitegrid", palette="pastel", color_codes=True)
            sns.set_context("talk")
            plt.figure(figsize=(2 * len(order), 2 * len(order)))
            ax = sns.heatmap(
                adjusted_pvalues
                <= 0.05,  # ToDo: When effects has been implemented, set this to: effects
                annot=adjusted_pvalues.round(3),  # ToDo: When effects has been
                # implemented, set this to: np.fix(-np.log10(adjusted_pvalues))
                yticklabels=order,
                xticklabels=order,
                cbar=False,  # ToDo: When effects has been implemented, set this to: True
                cbar_kws={"shrink": 0.7},
                square=True,
            )
            ax.xaxis.set_ticks_position("top")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
            ax.tick_params(axis="both", which="both", length=0)
            plt.tight_layout()
            plt.savefig(f"stats-{name}-{timestamp}.png")
            plt.close()
            save_obj(
                dict(
                    effects=effects,
                    pvalues=pvalues,
                    adjusted_pvalues=adjusted_pvalues,
                    order=order,
                ),
                f"stats-{name}-{timestamp}.pkl",
            )

    # create SHAP summary csv and figures
    if gen_shap:
        reg_metrics = [
            "explained_variance_score",
            "max_error",
            "mean_absolute_error",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
            "mean_poisson_deviance",
            "mean_gamma_deviance",
        ]
        if any([True for x in metrics if x in reg_metrics]):
            gen_report_shap_regres(
                results, output_dir=output_dir, plot_top_n_shap=plot_top_n_shap
            )
        else:
            gen_report_shap_class(
                results, output_dir=output_dir, plot_top_n_shap=plot_top_n_shap
            )
