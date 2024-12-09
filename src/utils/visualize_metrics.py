import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
        
def visualize_metrics(nb_metrics: dict, bert_metrics: dict, output_dir: str) -> None:
    """
    Generate comparison plots for Naive Bayes and BERT metrics.

    Args:
        nb_metrics (dict): Metrics for Naive Bayes.
        bert_metrics (dict): Metrics for BERT.
        output_dir (str): Directory to save the plots.

    Returns:
        None. Saves plots to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    models = ['Naive Bayes', 'BERT']
    metrics_df = pd.DataFrame({
        'Model': ['Naive Bayes'] * len(nb_metrics) + ['BERT'] * len(bert_metrics),
        'Label': list(nb_metrics.keys()) + list(bert_metrics.keys()),
        'Precision': [nb_metrics[label]['precision'] for label in nb_metrics] + 
                    [bert_metrics[label]['precision'] for label in bert_metrics],
        'Recall': [nb_metrics[label]['recall'] for label in nb_metrics] +
                [bert_metrics[label]['recall'] for label in bert_metrics],
        'F1-Score': [nb_metrics[label]['f1-score'] for label in nb_metrics] +
                    [bert_metrics[label]['f1-score'] for label in bert_metrics],
    })

    for metric in ['Precision', 'Recall', 'F1-Score']:
        plt.figure(figsize=(8, 6))
        for model in models:
            subset = metrics_df[metrics_df['Model'] == model]
            plt.plot(subset['Label'], subset[metric], marker='o', label=model)
        plt.title(f"{metric} Comparison by Model")
        plt.xlabel("Label")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{metric.lower()}_comparison.png"))
        plt.close()
