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
