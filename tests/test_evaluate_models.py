import os
import unittest
import pandas as pd
from unittest.mock import MagicMock
from src.utils.evaluate_models import load_test_data, evaluate_model
from src.models.naive_bayes_model import NaiveBayesModel


class TestEvaluateModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_path = "tests/mock_test_data.csv"
        cls.output_dir = "tests/mock_evaluation_results"

        # Create mock test data
        test_data = pd.DataFrame({
            'Message': ['This is a spam message', 'Hello, how are you?'],
            'Spam/Ham': [1, 0]
        })
        test_data.to_csv(cls.test_data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.test_data_path):
            os.remove(cls.test_data_path)
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_load_test_data(self):
        df = load_test_data(self.test_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('Message', df.columns)
        self.assertIn('Spam/Ham', df.columns)

    def test_evaluate_model(self):
        # Mock NaiveBayesModel
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0]  # Mock predictions

        test_data = load_test_data(self.test_data_path)
        evaluate_model(mock_model, test_data, self.output_dir)

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "MagicMock_report.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "MagicMock_confusion_matrix.csv")))


if __name__ == "__main__":
    unittest.main()
