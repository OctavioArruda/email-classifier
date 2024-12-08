import os
import pandas as pd
from src.models.naive_bayes_model import NaiveBayesModel
from src.models.bert_model import BERTModel


def load_unseen_data(file_path: str) -> pd.DataFrame:
    """
    Load unseen data for prediction.

    Args:
        file_path (str): Path to the CSV file containing unseen data.

    Returns:
        pd.DataFrame: DataFrame with the unseen data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def predict_with_model(model, messages: pd.Series) -> pd.DataFrame:
    """
    Generate predictions using a specified model.

    Args:
        model (object): Trained model (e.g., NaiveBayesModel or BERTModel).
        messages (pd.Series): Series containing message texts for prediction.

    Returns:
        pd.DataFrame: DataFrame with messages and their predicted labels.
    """
    predictions = model.predict(messages)
    return pd.DataFrame({
        'Message': messages,
        'Predicted Label': predictions
    })


def save_predictions_to_csv(predictions: pd.DataFrame, output_path: str) -> None:
    """
    Save predictions to a CSV file.

    Args:
        predictions (pd.DataFrame): DataFrame containing messages and predicted labels.
        output_path (str): File path to save the predictions.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # File paths
    unseen_data_path = "data/unseen/unseen_messages.csv"
    output_dir = "predictions"
    nb_output_path = os.path.join(output_dir, "naive_bayes_predictions.csv")
    bert_output_path = os.path.join(output_dir, "bert_predictions.csv")

    # Load unseen data
    unseen_data = load_unseen_data(unseen_data_path)

    # Predict using Naive Bayes Model
    print("Generating predictions with Naive Bayes model...")
    nb_model = NaiveBayesModel()
    nb_model.load("models/naive_bayes_model.pkl")
    nb_predictions = predict_with_model(nb_model, unseen_data['Message'])
    save_predictions_to_csv(nb_predictions, nb_output_path)

    # Predict using BERT Model
    print("Generating predictions with BERT model...")
    bert_model = BERTModel()
    bert_model.load("models/bert_model")
    bert_predictions = predict_with_model(bert_model, unseen_data['Message'])
    save_predictions_to_csv(bert_predictions, bert_output_path)
