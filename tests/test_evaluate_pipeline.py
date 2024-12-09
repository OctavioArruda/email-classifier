import unittest
import os
import pandas as pd
from src.utils.evaluate_pipeline import consolidate_metrics, plot_metrics_comparison


class TestEvaluatePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb_metrics = {"Ham": {"f1-score": 0.9}, "Spam": {"f1-score": 0.8}}
        cls.bert_metrics = {"Ham": {"f1-score": 0.95}, "Spam": {"f1-score": 0.85}}
        cls.output_dir = "tests/mock_evaluation_results"
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_consolidate_metrics(self):
        consolidated_df = consolidate_metrics(self.nb_metrics, self.bert_metrics)
        self.assertIsInstance(consolidated_df, pd.DataFrame)
        self.assertEqual(len(consolidated_df), 2)
        self.assertIn("Label", consolidated_df.columns)
        self.assertIn("Naive Bayes F1", consolidated_df.columns)
        self.assertIn("BERT F1", consolidated_df.columns)

    def test_plot_metrics_comparison(self):
        consolidated_df = consolidate_metrics(self.nb_metrics, self.bert_metrics)
        plot_metrics_comparison(consolidated_df, self.output_dir)
        plot_path = os.path.join(self.output_dir, "metrics_comparison.png")
        self.assertTrue(os.path.exists(plot_path))


if __name__ == "__main__":
    unittest.main()
