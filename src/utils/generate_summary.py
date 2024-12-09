import os
import pandas as pd

def generate_summary(metrics_df: pd.DataFrame, output_path: str) -> None:
    """
    Generate a summary report from the evaluation metrics and save it as a CSV.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics for various models.
                                   Expected columns: ['Model', 'Label', 'Precision', 'Recall', 'F1-Score'].
        output_path (str): File path to save the summary CSV.

    Returns:
        None. The summary is saved as a CSV file.
    """
    if not isinstance(metrics_df, pd.DataFrame):
        raise ValueError("`metrics_df` must be a pandas DataFrame.")

    required_columns = {"Model", "Label", "Precision", "Recall", "F1-Score"}
    if not required_columns.issubset(metrics_df.columns):
        raise ValueError(f"DataFrame must include columns: {required_columns}")

    # Group metrics by model and calculate overall averages
    summary_df = (
        metrics_df.groupby("Model")[["Precision", "Recall", "F1-Score"]]
        .mean()
        .reset_index()
    )
    summary_df.to_csv(output_path, index=False)
    print(f"Summary report saved to {output_path}")
