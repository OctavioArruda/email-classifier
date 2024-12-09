import os
import unittest
import pandas as pd
from src.utils.generate_summary import generate_summary

class TestGenerateSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_path = "tests/mock_summary.csv"
        cls.sample_data = pd.DataFrame({
            "Model": ["Naive Bayes", "Naive Bayes", "BERT", "BERT"],
            "Label": ["Ham", "Spam", "Ham", "Spam"],
            "Precision": [0.9, 0.8, 0.95, 0.85],
            "Recall": [0.92, 0.81, 0.96, 0.87],
            "F1-Score": [0.91, 0.8, 0.955, 0.86],
        })

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.output_path):
            os.remove(cls.output_path)

    def test_generate_summary_creates_csv(self):
        generate_summary(self.sample_data, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

    def test_generate_summary_content(self):
        generate_summary(self.sample_data, self.output_path)
        summary_df = pd.read_csv(self.output_path)
        self.assertIn("Model", summary_df.columns)
        self.assertIn("Precision", summary_df.columns)
        self.assertIn("Recall", summary_df.columns)
        self.assertIn("F1-Score", summary_df.columns)

    def test_invalid_dataframe_raises_error(self):
        with self.assertRaises(ValueError):
            generate_summary([], self.output_path)

    def test_missing_columns_raises_error(self):
        invalid_df = pd.DataFrame({"WrongColumn": [1, 2, 3]})
        with self.assertRaises(ValueError):
            generate_summary(invalid_df, self.output_path)


if __name__ == "__main__":
    unittest.main()
