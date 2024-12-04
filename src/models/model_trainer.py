import os
from src.models.bert_model import BERTModel
from src.models.naive_bayes_model import NaiveBayesModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class ModelTrainer:
    def __init__(self, model_type="bert"):
        """
        Initializes the trainer with the specified model type.
        :param model_type: "bert" or "naive_bayes".
        """
        self.model_type = model_type.lower()
        if self.model_type == "bert":
            self.model = BERTModel()
        elif self.model_type == "naive_bayes":
            self.model = NaiveBayesModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, texts, labels):
        """
        Train the model on the given dataset.
        :param texts: List of text inputs.
        :param labels: Corresponding labels.
        """
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        self.model.train(
            {"text": train_texts, "label": train_labels},
            {"text": val_texts, "label": val_labels},
        )

    def evaluate(self, texts, labels):
        """
        Evaluate the model on the given dataset.
        :param texts: List of text inputs.
        :param labels: Corresponding labels.
        :return: A classification report.
        """
        accuracy, report = self.model.evaluate({"text": texts, "label": labels})
        print("Classification Report:")
        print(report)
        return accuracy, report

    def save_model(self, output_dir="models/"):
        """
        Save the trained model.
        :param output_dir: Directory to save the model.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)


if __name__ == "__main__":
    # Example usage
    texts = [
        "This is a spam email.",
        "This is a regular email.",
        "Get rich quick!",
        "Meeting scheduled for tomorrow.",
    ]
    labels = [1, 0, 1, 0]

    trainer = ModelTrainer(model_type="bert")
    trainer.train(texts, labels)
    trainer.evaluate(texts, labels)
    trainer.save_model()
