import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate plots for evaluation metrics and save them to the specified directory.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics for various models.
                                   Expected columns: ['Model', 'Label', 'Precision', 'Recall', 'F1-Score'].
        output_dir (str): Directory to save the generated plots.

    Returns:
        None. Plots are saved to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = metrics_df['Label'].unique()
    models = metrics_df['Model'].unique()

    for metric in ['Precision', 'Recall', 'F1-Score']:
        plt.figure(figsize=(8, 6))
        for model in models:
            subset = metrics_df[metrics_df['Model'] == model]
            plt.plot(labels, subset[metric], marker='o', label=model)

        plt.title(f"{metric} Comparison by Model")
        plt.xlabel("Labels")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        output_path = os.path.join(output_dir, f"{metric.lower()}_comparison.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved {metric} plot to {output_path}")
