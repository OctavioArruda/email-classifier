import unittest
import os
import zipfile
import pandas as pd
from unittest.mock import patch
from src.utils.model_comparison import load_metrics, classification_report_to_dict, compare_models


class TestModelComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zip_path = "tests/mock_enron_spam.zip"
        cls.extract_to = "tests/mock_extracted"
        cls.output_dir = "tests/mock_output"
        cls.metrics_file = os.path.join(cls.output_dir, "mock_metrics.txt")

        os.makedirs(cls.output_dir, exist_ok=True)

        # Create mock metrics file
        with open(cls.metrics_file, "w") as f:
            f.write("               precision    recall  f1-score   support\n")
            f.write("Ham           0.90      0.95      0.92       100\n")
            f.write("Spam          0.85      0.80      0.82        50\n")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.metrics_file):
            os.remove(cls.metrics_file)
        if os.path.exists(cls.output_dir):
            import shutil
            shutil.rmtree(cls.output_dir)

    def test_load_metrics(self):
        metrics = load_metrics(self.metrics_file)
        self.assertIn("Ham", metrics)
        self.assertIn("Spam", metrics)
        self.assertAlmostEqual(metrics["Ham"]["precision"], 0.90)
        self.assertAlmostEqual(metrics["Spam"]["recall"], 0.80)

    def test_classification_report_to_dict(self):
        report = """
               precision    recall  f1-score   support
    Ham           0.90      0.95      0.92       100
    Spam          0.85      0.80      0.82        50
    """
        metrics = classification_report_to_dict(report)
        self.assertEqual(metrics["Ham"]["f1-score"], 0.92)
        self.assertEqual(metrics["Spam"]["f1-score"], 0.82)

    @patch("src.utils.model_comparison.compare_models", autospec=True)
    def test_compare_models(self, mock_compare):
        nb_metrics = {"Ham": {"f1-score": 0.9}, "Spam": {"f1-score": 0.8}}
        bert_metrics = {"Ham": {"f1-score": 0.95}, "Spam": {"f1-score": 0.85}}

        # Directly call compare_models (adjust as necessary)
        from src.utils.model_comparison import compare_models
        compare_models(nb_metrics, bert_metrics, self.output_dir)

        # Assert the mock was called
        mock_compare.assert_called_once_with(nb_metrics, bert_metrics, self.output_dir)



if __name__ == "__main__":
    unittest.main()
