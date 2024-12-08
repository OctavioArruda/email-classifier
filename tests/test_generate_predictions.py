import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.utils.generate_predictions import (
    load_unseen_data,
    predict_with_model,
    save_predictions_to_csv
)


class TestGeneratePredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_csv = "tests/mock_unseen_data.csv"
        cls.output_csv = "tests/mock_predictions.csv"

        # Mock unseen data
        pd.DataFrame({'Message': ['Buy now!', 'How are you?']}).to_csv(cls.input_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        import os
        if os.path.exists(cls.input_csv):
            os.remove(cls.input_csv)
        if os.path.exists(cls.output_csv):
            os.remove(cls.output_csv)

    def test_load_unseen_data(self):
        df = load_unseen_data(self.input_csv)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Message', df.columns)
        self.assertEqual(len(df), 2)

    @patch("src.models.naive_bayes_model.NaiveBayesModel")
    def test_predict_with_model(self, mock_model_class):
        mock_model = mock_model_class.return_value
        mock_model.predict.return_value = [1, 0]

        messages = pd.Series(['Buy now!', 'How are you?'])
        predictions = predict_with_model(mock_model, messages)

        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn('Message', predictions.columns)
        self.assertIn('Predicted Label', predictions.columns)
        self.assertEqual(len(predictions), len(messages))

    def test_save_predictions_to_csv(self):
        predictions = pd.DataFrame({
            'Message': ['Buy now!', 'How are you?'],
            'Predicted Label': [1, 0]
        })
        save_predictions_to_csv(predictions, self.output_csv)

        self.assertTrue(os.path.exists(self.output_csv))
        saved_data = pd.read_csv(self.output_csv)
        pd.testing.assert_frame_equal(saved_data, predictions)


if __name__ == "__main__":
    unittest.main()
