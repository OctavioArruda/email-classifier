import os
import pandas as pd
import matplotlib.pyplot as plt


def consolidate_metrics(nb_metrics: dict, bert_metrics: dict) -> pd.DataFrame:
    """
    Consolidate metrics from Naive Bayes and BERT into a single DataFrame.

    Args:
        nb_metrics (dict): Metrics from Naive Bayes.
        bert_metrics (dict): Metrics from BERT.

    Returns:
        pd.DataFrame: Consolidated metrics for analysis.
    """
    labels = list(nb_metrics.keys())
    consolidated = {
        "Label": labels,
        "Naive Bayes F1": [nb_metrics[label]["f1-score"] for label in labels],
        "BERT F1": [bert_metrics[label]["f1-score"] for label in labels],
    }
    return pd.DataFrame(consolidated)


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create a bar plot comparing F1-scores for Naive Bayes and BERT.

    Args:
        metrics_df (pd.DataFrame): Consolidated metrics.
        output_dir (str): Directory to save the plot.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = metrics_df["Label"]
    x = range(len(labels))
    width = 0.35

    plt.bar(x, metrics_df["Naive Bayes F1"], width, label="Naive Bayes")
    plt.bar([p + width for p in x], metrics_df["BERT F1"], width, label="BERT")

    plt.xlabel("Labels")
    plt.ylabel("F1-Score")
    plt.title("Model Performance Comparison")
    plt.xticks([p + width / 2 for p in x], labels)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()
    