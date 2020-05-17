import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gen_report(results, prefix, metrics):
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
