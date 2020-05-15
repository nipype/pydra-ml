import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gen_report(results, prefix):
    df = pd.DataFrame(columns=["AUC score", "Classifier", "type"])
    for val in results:
        auc = val[1].output.auc
        if not isinstance(auc, list):
            auc = [auc]
        name = val[0][prefix + ".clf_info"][1].split("Classifier")[0]
        permute = val[0][prefix + ".permute"]
        for aucval in auc:
            df = df.append(
                {
                    "AUC score": aucval if aucval is not None else np.nan,
                    "Classifier": name,
                    "type": "null" if permute else "data",
                },
                ignore_index=True,
            )
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    sns.set_context("talk")
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        x="Classifier",
        y="AUC score",
        hue="type",
        data=df,
        split=True,
        inner="quartile",
        hue_order=["data", "null"],
        order=[group[0] for group in df.groupby("Classifier")],
    )
    sns.despine(left=True)
    plt.savefig("test.png")
