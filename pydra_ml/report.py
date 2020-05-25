import warnings
import datetime
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def plot_summary(
    summary,
    output_dir=None,
    filename="shap_LogisticRegression_all_predictions",
    plot_top_n_shap=16,
):
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
    return


def shaps_to_summary(
    shaps_n_splits,
    feature_names=None,
    output_dir=None,
    filename="shap_LogisticRegression_all_predictions",
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
    return


def gen_report_shap(results, output_dir="./", plot_top_n_shap=16):
    # Create shap_dir
    timestamp = datetime.datetime.utcnow().isoformat()
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
    return


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
            order=[group[0] for group in df.groupby("Classifier")],
        )
        ax.set_ylabel(name)
        sns.despine(left=True)
        import datetime

        timestamp = datetime.datetime.utcnow().isoformat()
        plt.savefig(f"test-{name}-{timestamp}.png")

    # create SHAP summary csv and figures
    if gen_shap:
        gen_report_shap(results, output_dir=output_dir, plot_top_n_shap=plot_top_n_shap)
