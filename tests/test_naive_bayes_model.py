import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.models.naive_bayes_model import NaiveBayesModel


class TestNaiveBayesModel(unittest.TestCase):
    def setUp(self):
        # Initialize the NaiveBayesModel
        self.model = NaiveBayesModel()

    @patch.object(CountVectorizer, 'fit_transform')
    @patch.object(MultinomialNB, 'fit')
    def test_train(self, mock_fit, mock_fit_transform):
        # Mock the vectorizer and model
        mock_fit_transform.return_value = MagicMock()  # Simulated sparse matrix
        mock_fit.return_value = None

        # Create mock training data
        train_data = pd.DataFrame({
            'text': ['This is spam', 'This is ham'],
            'label': [1, 0]
        })

        # Call train
        self.model.train(train_data)

        # Assert vectorizer and model methods are called
        mock_fit_transform.assert_called_once_with(train_data['text'])
        mock_fit.assert_called_once()

    @patch.object(CountVectorizer, 'transform')
    @patch.object(MultinomialNB, 'predict')
    @patch('src.models.naive_bayes_model.accuracy_score')
    @patch('src.models.naive_bayes_model.classification_report')
    def test_evaluate(self, mock_report, mock_accuracy, mock_predict, mock_transform):
        # Mock vectorizer, model, and metrics
        mock_transform.return_value = MagicMock()  # Simulated sparse matrix
        mock_predict.return_value = [1, 0]
        mock_accuracy.return_value = 0.95
        mock_report.return_value = "Mocked Classification Report"

        # Create mock evaluation data
        eval_data = pd.DataFrame({
            'text': ['This is spam', 'This is ham'],
            'label': [1, 0]
        })

        # Call evaluate
        accuracy, report = self.model.evaluate(eval_data)

        # Assert vectorizer, model, and metric methods are called
        mock_transform.assert_called_once_with(eval_data['text'])
        mock_predict.assert_called_once()
        mock_accuracy.assert_called_once_with(eval_data['label'], mock_predict.return_value)
        mock_report.assert_called_once_with(eval_data['label'], mock_predict.return_value)

        # Check the returned accuracy and report
        self.assertEqual(accuracy, 0.95)
        self.assertEqual(report, "Mocked Classification Report")


if __name__ == "__main__":
    unittest.main()
