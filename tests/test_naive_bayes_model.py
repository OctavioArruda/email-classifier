import unittest
import pickle
from unittest.mock import mock_open, MagicMock, patch
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
        """
        Test the train method of NaiveBayesModel.
        """
        # Mock the vectorizer and model
        mock_fit_transform.return_value = MagicMock()  # Simulated sparse matrix
        mock_fit.return_value = None

        # Create mock training data as a DataFrame
        train_data = pd.DataFrame({
            'text': ['This is spam', 'This is ham'],
            'label': [1, 0]
        })

        # Call train
        self.model.train(train_data)

        # Convert to list for assertion
        expected_text = train_data['text'].tolist()

        # Assertions to ensure fit_transform and fit were called
        mock_fit_transform.assert_called_once_with(expected_text)
        mock_fit.assert_called_once()


    @patch.object(CountVectorizer, 'transform')
    @patch.object(MultinomialNB, 'predict')
    @patch('src.models.naive_bayes_model.accuracy_score')
    @patch('src.models.naive_bayes_model.classification_report')
    def test_evaluate(self, mock_report, mock_accuracy, mock_predict, mock_transform):
        """
        Test the evaluate method of NaiveBayesModel.
        """
        # Mock the vectorizer and model as loaded
        self.model.vectorizer = MagicMock(spec=CountVectorizer)
        self.model.model = MagicMock(spec=MultinomialNB)

        # Mock the methods and return values
        mock_transform.return_value = MagicMock()  # Simulated sparse matrix
        mock_predict.return_value = [1, 0]
        mock_accuracy.return_value = 0.95
        mock_report.return_value = "Mocked Classification Report"

        # Create mock evaluation data
        eval_data = {'text': ['This is spam', 'This is ham'], 'label': [1, 0]}

        # Call evaluate
        accuracy, report = self.model.evaluate(eval_data)

        # Assertions
        assert accuracy == 0.95
        assert report == "Mocked Classification Report"

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    def test_load_method(self, mock_pickle_load, mock_open):
        # Mock the pickle load to return fake model and vectorizer
        mock_model = MagicMock()
        mock_vectorizer = MagicMock()
        mock_pickle_load.side_effect = [mock_model, mock_vectorizer]

        naive_bayes = NaiveBayesModel()
        naive_bayes.load("test_dir")

        # Verify the files were opened
        mock_open.assert_any_call("test_dir/naive_bayes_model.pkl", "rb")
        mock_open.assert_any_call("test_dir/vectorizer.pkl", "rb")

        # Verify the model and vectorizer were set
        self.assertEqual(naive_bayes.model, mock_model)
        self.assertEqual(naive_bayes.vectorizer, mock_vectorizer)

    def test_predict_without_loading(self):
        naive_bayes = NaiveBayesModel()
        with self.assertRaises(ValueError) as context:
            naive_bayes.predict("Some text")
        self.assertIn("Model or vectorizer not loaded", str(context.exception))


    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    def test_predict_method(self, mock_pickle_load, mock_open):
        # Mock model and vectorizer
        mock_model = MagicMock()
        mock_model.predict.return_value = ["spam"]
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = "vectorized_text"
        mock_pickle_load.side_effect = [mock_model, mock_vectorizer]

        naive_bayes = NaiveBayesModel()
        naive_bayes.load("test_dir")

        # Test prediction
        prediction = naive_bayes.predict("Some text")
        self.assertEqual(prediction, "spam")
        mock_vectorizer.transform.assert_called_once_with(["Some text"])
        mock_model.predict.assert_called_once_with("vectorized_text")
        
if __name__ == "__main__":
    unittest.main()
