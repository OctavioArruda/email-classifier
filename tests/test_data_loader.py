import unittest
from unittest.mock import patch
import pandas as pd
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.file_path = "data/enron_spam_dataset.csv"
        self.mock_data = pd.DataFrame({
            "text": ["Sample email text"] * 10,  # Create 10 identical rows
            "label": [0, 1] * 5  # Alternating labels
        })

    @patch("pandas.read_csv")
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = self.mock_data
        loader = DataLoader(self.file_path, model_type="naive_bayes")
        data = loader.load_data()

        self.assertEqual(len(data), 10)  # Check if all rows are loaded
        self.assertIn("tokens", data.columns)  # Ensure 'tokens' column exists

    @patch("pandas.read_csv")
    def test_split_data(self, mock_read_csv):
        mock_read_csv.return_value = self.mock_data
        loader = DataLoader(self.file_path, model_type="naive_bayes")
        data = loader.load_data()

        # Split the data
        train_data, val_data, test_data = loader.split_data(data)

        # Assert sizes of the splits
        self.assertEqual(len(train_data), 7)  # 70% of 10
        self.assertEqual(len(val_data), 1)    # 15% of 10
        self.assertEqual(len(test_data), 2)  # Remaining 15% of 10

if __name__ == "__main__":
    unittest.main()
