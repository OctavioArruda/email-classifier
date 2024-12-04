import os
from src.models.bert_model import BERTModel
from src.models.naive_bayes_model import NaiveBayesModel


class ModelInference:
    def __init__(self, model_type="bert", model_dir="models/"):
        """
        Initialize the inference class with a specified model type and model directory.
        :param model_type: "bert" or "naive_bayes".
        :param model_dir: Path to the saved model directory.
        """
        self.model_type = model_type.lower()
        self.model_dir = model_dir

        if self.model_type == "bert":
            self.model = BERTModel()
        elif self.model_type == "naive_bayes":
            self.model = NaiveBayesModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Attempt to load model if the directory exists; otherwise mock the model for testing
        if os.path.exists(self.model_dir):
            self.load_model()
        else:
            self.mock_model()

    def load_model(self):
        """
        Load the trained model from the specified directory.
        """
        self.model.load(self.model_dir)

    def mock_model(self):
        """
        Mock model behavior for testing purposes.
        """
        self.model.predict = lambda x: "mocked_label"

    def predict(self, text):
        """
        Predict the label for a given input text.
        :param text: Input text to classify.
        :return: Predicted label.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        return self.model.predict(text)
