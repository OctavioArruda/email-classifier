import os
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    with open(file_path, 'r') as f:
        report = f.read()
    return classification_report_to_dict(report)


def classification_report_to_dict(report: str) -> dict:
    lines = report.strip().splitlines()
    results = {}
    for line in lines[1:]:  # Skip headers
        row = line.strip().split()
        if len(row) >= 5:  # Ensure valid metrics line
            label = row[0]
            results[label] = {
                "precision": float(row[1]),
                "recall": float(row[2]),
                "f1-score": float(row[3]),
                "support": int(row[4]),
            }
    return results


def compare_models(nb_metrics: dict, bert_metrics: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    labels = list(nb_metrics.keys())
    nb_f1 = [nb_metrics[label]["f1-score"] for label in labels]
    bert_f1 = [bert_metrics[label]["f1-score"] for label in labels]

    x = range(len(labels))
    plt.bar(x, nb_f1, width=0.4, label="Naive Bayes", align="center")
    plt.bar([p + 0.4 for p in x], bert_f1, width=0.4, label="BERT", align="center")
    plt.xticks([p + 0.2 for p in x], labels)
    plt.xlabel("Class")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Comparison: Naive Bayes vs. BERT")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"))
    plt.close()
