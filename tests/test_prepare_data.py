import os
import unittest
import pandas as pd
import zipfile
from unittest.mock import patch
from src.utils.prepare_data import extract_dataset, load_data, preprocess_data, split_data, save_splits


class TestPrepareData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zip_path = "tests/mock_enron_spam.zip"
        cls.extract_to = "tests/mock_extracted"
        cls.output_dir = "tests/mock_output"  # Define the output directory

        os.makedirs(cls.extract_to, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)  # Create the output directory

        # Create a mock CSV file
        mock_csv_path = os.path.join(cls.extract_to, "enron_spam_data.csv")
        data = {"Message": ["msg1", "msg2"], "Spam/Ham": ["ham", "spam"]}
        pd.DataFrame(data).to_csv(mock_csv_path, index=False)

        # Create a ZIP file containing the CSV
        with zipfile.ZipFile(cls.zip_path, "w") as zipf:
            zipf.write(mock_csv_path, arcname="enron_spam_data.csv")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.extract_to):
            shutil.rmtree(cls.extract_to)
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    def test_extract_dataset(self):
        csv_path = extract_dataset(self.zip_path, self.extract_to)
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(csv_path.endswith(".csv"))

    def test_load_data(self):
        mock_csv_path = os.path.join(self.extract_to, "enron_spam_data.csv")
        df = load_data(mock_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Message", df.columns)
        self.assertIn("Spam/Ham", df.columns)

    def test_preprocess_data(self):
        data = pd.DataFrame({
            "Message": ["<p>Hello World!</p>", "  Buy   NOW  !"],
            "Spam/Ham": ["ham", "spam"]
        })
        processed = preprocess_data(data)
        self.assertEqual(processed["Message"].iloc[0], "hello world!")
        self.assertEqual(processed["Message"].iloc[1], "buy now !")

    def test_split_data(self):
        data = pd.DataFrame({
            "Message": ["msg1", "msg2", "msg3", "msg4"],
            "Spam/Ham": ["ham", "spam", "ham", "spam"]
        })
        train, val, test = split_data(data, test_size=0.25, random_state=42)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)

    def test_save_splits(self):
        data = pd.DataFrame({
            "Message": ["msg1", "msg2", "msg3", "msg4"],
            "Spam/Ham": ["ham", "spam", "ham", "spam"]
        })
        train, val, test = split_data(data, test_size=0.2, random_state=42)
        save_splits(train, val, test, self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "val.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test.csv")))


if __name__ == "__main__":
    unittest.main()
