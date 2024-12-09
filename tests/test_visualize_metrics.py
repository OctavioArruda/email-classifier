import os
import unittest
import pandas as pd
from src.utils.visualize_metrics import plot_metrics

class TestVisualizeMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = "tests/mock_visualizations"
        os.makedirs(cls.output_dir, exist_ok=True)
        cls.sample_data = pd.DataFrame({
            "Model": ["Naive Bayes", "Naive Bayes", "BERT", "BERT"],
            "Label": ["Ham", "Spam", "Ham", "Spam"],
            "Precision": [0.9, 0.8, 0.95, 0.85],
            "Recall": [0.92, 0.81, 0.96, 0.87],
            "F1-Score": [0.91, 0.8, 0.955, 0.86],
        })

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_plot_metrics_creates_plots(self):
        plot_metrics(self.sample_data, self.output_dir)
        for metric in ["precision", "recall", "f1-score"]:
            plot_path = os.path.join(self.output_dir, f"{metric}_comparison.png")
            self.assertTrue(os.path.exists(plot_path))

    def test_invalid_dataframe_raises_error(self):
        invalid_data = {"Model": ["NB"], "Label": ["Ham"], "F1-Score": [0.9]}
        with self.assertRaises(KeyError):
            plot_metrics(pd.DataFrame(invalid_data), self.output_dir)


if __name__ == "__main__":
    unittest.main()
