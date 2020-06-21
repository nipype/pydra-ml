import warnings
import datetime
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def plot_summary(summary, output_dir=None, filename="shap_plot", plot_top_n_shap=16):
    plt.clf()
    plt.figure(figsize=(8, 12))
    # plot without all bootstrapping values
    summary = summary[["mean", "std", "min", "max"]]
    num_features = len(list(summary.index))
    if (plot_top_n_shap != 1 and type(plot_top_n_shap) == float) or type(
        plot_top_n_shap
    ) == int:
        # if plot_top_n_shap != 1.0 but includes 1 (int)
        if plot_top_n_shap <= 0:
            raise ValueError(
                "plot_top_n_shap should be a float between 0 and 1.0 or an integer >= 1. You set to zero or negative."
            )
        elif plot_top_n_shap < 1:
            plot_top_n_shap = int(np.round(plot_top_n_shap * num_features))
        summary = summary.iloc[:plot_top_n_shap, :]
        filename += f"_top_{plot_top_n_shap}"

    hm = sns.heatmap(
        summary.round(3), annot=True, xticklabels=True, yticklabels=True, cbar=False
    )
    hm.set_xticklabels(summary.columns, rotation=45)
    hm.set_yticklabels(summary.index, rotation=0)
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(output_dir + f"summary_{filename}.png", dpi=100)


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
    # add summary stats
    shaps_n_splits["mean"] = shaps_n_splits.mean(axis=1)
    shaps_n_splits["std"] = shaps_n_splits.std(axis=1)
    shaps_n_splits["min"] = shaps_n_splits.min(axis=1)
    shaps_n_splits["max"] = shaps_n_splits.max(axis=1)
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

    for model_results in results:
        model_name = model_results[0].get("ml_wf.clf_info")[1]
        indexes_all[model_name] = []
        shaps = model_results[
            1
        ].output.shaps  # this is (N, P, F) N splits, P predictions, F feature_names
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
        }  # this is key with shape (F, N) where F is feature_names, N is mean shap values across splits
        # Obtain values for each bootstrapping split, then append summary statistics to shaps_n_splits
        for split_i in range(n_splits):
            shaps_i = shaps[split_i]  # all shap values for this bootstrapping split
            y_true = y_true_and_preds[split_i][0]
            y_pred = y_true_and_preds[split_i][1]
            # split_performance = accuracy_score(y_true, y_pred)
            split_performance = explained_variance_score(y_true, y_pred)

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

            #  For each quadrant, obtain F shap values for P predictions, take the absolute mean weighted by performance across all predictions
            for quadrant in ["tp", "tn", "fp", "fn"]:
                if len(indexes.get(quadrant)) == 0:
                    warnings.warn(
                        f"There were no {quadrant.upper()}s, this will output NaNs in the csv and figure for this split column"
                    )
                shaps_i_quadrant = shaps_i[
                    indexes.get(quadrant)
                ]  # shape (P, F) P prediction x F feature_names
                abs_weighted_shap_values = np.abs(shaps_i_quadrant) * split_performance
                shaps_n_splits[quadrant].append(
                    np.mean(abs_weighted_shap_values, axis=0)
                )
            #  obtain F shap values for P predictions, take the absolute mean weighted by performance across all predictions
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

    for model_results in results:
        model_name = model_results[0].get("ml_wf.clf_info")[1]
        indexes_all[model_name] = []
        shaps = model_results[
            1
        ].output.shaps  # this is (N, P, F) N splits, P predictions, F feature_names
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
        }  # this is key with shape (F, N) where F is feature_names, N is mean shap values across splits
        # Obtain values for each bootstrapping split, then append summary statistics to shaps_n_splits
        for split_i in range(n_splits):
            shaps_i = shaps[split_i]  # all shap values for this bootstrapping split
            y_true = y_true_and_preds[split_i][0]
            y_pred = y_true_and_preds[split_i][1]
            split_performance = explained_variance_score(y_true, y_pred)

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

            #  For each quadrant, obtain F shap values for P predictions, take the absolute mean weighted by performance across all predictions
            for quadrant in ["lp", "lm", "um", "up"]:
                if len(indexes.get(quadrant)) == 0:
                    warnings.warn(
                        f"There were no {quadrant.upper()}s, this will output NaNs in the csv and figure for this split column"
                    )
                shaps_i_quadrant = shaps_i[
                    indexes.get(quadrant)
                ]  # shape (P, F) P prediction x F feature_names
                abs_weighted_shap_values = np.abs(shaps_i_quadrant) * split_performance
                shaps_n_splits[quadrant].append(
                    np.mean(abs_weighted_shap_values, axis=0)
                )
            #  obtain F shap values for P predictions, take the absolute mean weighted by performance across all predictions
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


def compute_pairwise_stats(df):
    """Run Wilcoxon signed rank tests across pairs of classifiers.

    When comparing a classifier to itself, compare to its null distribution.
    A one sided test is used.

    Assumes that the dataframe has three keys: Classifier, type, and score
    with type referring to either the data distribution or the null distribution

    """
    N = len(df.Classifier.unique())
    effects = np.zeros((N, N)) * np.nan
    pvalues = np.zeros((N, N)) * np.nan
    for idx1, group1 in enumerate(df.groupby("Classifier")):
        filter = group1[1].apply(lambda x: x.type == "data", axis=1).values
        group1df = group1[1].iloc[filter, :]
        filter = group1[1].apply(lambda x: x.type == "null", axis=1).values
        group1nulldf = group1[1].iloc[filter, :]
        for idx2, group2 in enumerate(df.groupby("Classifier")):
            filter = group2[1].apply(lambda x: x.type == "data", axis=1).values
            group2df = group2[1].iloc[filter, :]
            if group1[0] != group2[0]:
                stat, pval = wilcoxon(
                    group1df["score"].values,
                    group2df["score"].values,
                    alternative="greater",
                )
            else:
                stat, pval = wilcoxon(
                    group1df["score"].values,
                    group1nulldf["score"].values,
                    alternative="greater",
                )
            effects[idx1, idx2] = stat
            pvalues[idx1, idx2] = pval
    return effects, pvalues


def gen_report(
    results, prefix, metrics, gen_shap=True, output_dir="./", plot_top_n_shap=16
):
    if len(results) == 0:
        raise ValueError("results is empty")
    df = pd.DataFrame(columns=["metric", "score", "Classifier", "type"])
    for val in results:
        score = val[1].output.score
        if not isinstance(score, list):
            score = [score]
        name = val[0][prefix + ".clf_info"][1].split("Classifier")[0]
        permute = val[0][prefix + ".permute"]
        for scoreval in score:
            for idx, metric in enumerate(metrics):
                df = df.append(
                    {
                        "Classifier": name,
                        "type": "null" if permute else "data",
                        "metric": metrics[idx],
                        "score": scoreval[idx] if scoreval[idx] is not None else np.nan,
                    },
                    ignore_index=True,
                )
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
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
        ax.set_ylabel(name)
        ax.legend(loc="center right", bbox_to_anchor=(1.2, 0.5), ncol=1)
        ax.tick_params(axis="both", which="both", length=0)
        sns.despine(left=True)
        plt.tight_layout()

        import datetime

        timestamp = datetime.datetime.utcnow().isoformat()
        timestamp = timestamp.replace(":", "").replace("-", "")
        plt.savefig(f"test-{name}-{timestamp}.png")
        plt.close()

        # Create comparison stats table if the metric is a score
        if "score" in name:
            effects, pvalues, = compute_pairwise_stats(subdf)
            sns.set(style="whitegrid", palette="pastel", color_codes=True)
            sns.set_context("talk")
            plt.figure(figsize=(2 * len(order), 2 * len(order)))
            # plt.figure(figsize=(8, 8))
            ax = sns.heatmap(
                effects,
                annot=np.fix(-np.log10(pvalues)),
                yticklabels=order,
                xticklabels=order,
                cbar=True,
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
                dict(effects=effects, pvalues=pvalues, order=order),
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
