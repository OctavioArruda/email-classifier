import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class NaiveBayesModel:
    def __init__(self):
        """
        Initialize the Naive Bayes model and vectorizer.
        Both are set to None initially, indicating the model is unloaded.
        """
        self.vectorizer = None
        self.model = None

    def train(self, train_data):
        """
        Train the Naive Bayes model on the given data.

        Parameters:
        - train_data (dict or pandas.DataFrame): A dictionary with 'text' and 'label' keys or a DataFrame.

        Raises:
        - ValueError: If training data is invalid or empty.
        """
        if isinstance(train_data, pd.DataFrame):
            # Ensure the DataFrame has the necessary columns
            if 'text' not in train_data.columns or 'label' not in train_data.columns:
                raise ValueError("Training data must contain 'text' and 'label' columns.")
            text = train_data['text'].tolist()
            labels = train_data['label'].tolist()
        elif isinstance(train_data, dict):
            # Ensure the dictionary has the necessary keys
            if 'text' not in train_data or 'label' not in train_data:
                raise ValueError("Training data must contain 'text' and 'label' keys.")
            text = train_data['text']
            labels = train_data['label']
        else:
            raise ValueError("Training data must be a dictionary or a pandas DataFrame.")

        # Initialize vectorizer and model
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

        # Vectorize and train the model
        X_train = self.vectorizer.fit_transform(text)
        self.model.fit(X_train, labels)



    def evaluate(self, data):
        """
        Evaluate the model on the given data.

        Parameters:
        - data (dict): A dictionary with 'text' (list of strings) and 'label' (list of labels).

        Returns:
        - tuple: A tuple containing accuracy (float) and a classification report (string).

        Raises:
        - ValueError: If the model or vectorizer is not loaded or data is invalid.
        """
        if not self.is_loaded():
            raise ValueError("Model or vectorizer not loaded.")
        if not data or 'text' not in data or 'label' not in data:
            raise ValueError("Evaluation data must contain 'text' and 'label' keys.")

        # Vectorize and predict
        X = self.vectorizer.transform(data['text'])
        y = data['label']
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report

    def load(self, model_dir):
        """
        Load the Naive Bayes model and vectorizer from a directory.

        Parameters:
        - model_dir (str): The directory containing the model and vectorizer files.

        Raises:
        - FileNotFoundError: If the model or vectorizer files are not found.
        """
        try:
            with open(f"{model_dir}/naive_bayes_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(f"{model_dir}/vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find model files in {model_dir}: {e}")

    def unload(self):
        """
        Unload the model and vectorizer, setting them to None.
        """
        self.model = None
        self.vectorizer = None

    def is_loaded(self):
        """
        Check if the model and vectorizer are loaded.

        Returns:
        - bool: True if both the model and vectorizer are loaded, False otherwise.
        """
        return self.model is not None and self.vectorizer is not None

    def predict(self, text):
        """
        Perform prediction on the given text.

        Parameters:
        - text (str): The input text to classify.

        Returns:
        - int: The predicted class label.

        Raises:
        - ValueError: If the model or vectorizer is not loaded or the vectorizer is not fitted.
        """
        if not self.is_loaded():
            raise ValueError("Model or vectorizer not loaded.")
        
        # Check if the vectorizer is fitted
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
            raise ValueError("Vectorizer is not fitted.")

        vectorized_text = self.vectorizer.transform([text])
        return self.model.predict(vectorized_text)[0]
