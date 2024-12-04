import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, train_data):
        # Vectorize the training data
        X_train = self.vectorizer.fit_transform(train_data['text'])
        y_train = train_data['label']
        self.model.fit(X_train, y_train)

    def evaluate(self, data):
        # Vectorize and predict
        X = self.vectorizer.transform(data['text'])
        y = data['label']
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report
    
    def load(self, model_dir):
        """
        Load Naive Bayes model and vectorizer from a directory.
        """
        with open(f"{model_dir}/naive_bayes_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(f"{model_dir}/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text):
        """
        Perform prediction on the given text.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer not loaded.")
        
        # Check if the vectorizer is fitted
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
            raise ValueError("Model or vectorizer not loaded.")

        vectorized_text = self.vectorizer.transform([text])
        return self.model.predict(vectorized_text)[0]

