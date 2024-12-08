import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.models.naive_bayes_model import NaiveBayesModel
from src.models.bert_model import BERTModel


def load_test_data(test_data_path: str) -> pd.DataFrame:
    """Load test dataset."""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    return pd.read_csv(test_data_path)


def evaluate_model(model, test_data: pd.DataFrame, output_dir: str) -> None:
    """Evaluate a given model and save results."""
    X_test = test_data['Message']
    y_test = test_data['Spam/Ham']

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, target_names=['Ham', 'Spam'])
    matrix = confusion_matrix(y_test, predictions)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model.__class__.__name__}_report.txt"), "w") as report_file:
        report_file.write(report)

    pd.DataFrame(matrix, columns=['Predicted Ham', 'Predicted Spam'],
                 index=['Actual Ham', 'Actual Spam']).to_csv(
        os.path.join(output_dir, f"{model.__class__.__name__}_confusion_matrix.csv"))

    print(f"{model.__class__.__name__} evaluation results saved to {output_dir}")


if __name__ == "__main__":
    # Paths
    test_data_path = "data/processed/test.csv"
    output_dir = "evaluation_results"

    # Load test data
    test_data = load_test_data(test_data_path)

    # Evaluate Naive Bayes
    nb_model = NaiveBayesModel()
    nb_model.load("models/naive_bayes_model.pkl")
    evaluate_model(nb_model, test_data, output_dir)

    # Evaluate BERT
    bert_model = BERTModel()
    bert_model.load("models/bert_model")
    evaluate_model(bert_model, test_data, output_dir)
