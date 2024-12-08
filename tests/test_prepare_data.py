import os
import unittest
import pandas as pd
from src.utils.prepare_data import extract_dataset, load_data, preprocess_data, split_data, save_splits


class TestPrepareData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.zip_path = os.path.join(cls.root_dir, "data/raw/enron_spam_data.zip")  # Updated file name
        cls.extract_to = os.path.join(cls.root_dir, "tests/temp")
        cls.output_dir = os.path.join(cls.root_dir, "tests/output")
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.extract_to):
            shutil.rmtree(cls.extract_to, ignore_errors=True)
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_extract_dataset(self):
        csv_path = extract_dataset(self.zip_path, self.extract_to)
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(csv_path.endswith(".csv"))

    def test_load_data(self):
        csv_path = extract_dataset(self.zip_path, self.extract_to)
        df = load_data(csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Message ID', df.columns)
        self.assertIn('Message', df.columns)
        self.assertIn('Spam/Ham', df.columns)
        self.assertGreater(len(df), 0)

    def test_preprocess_data(self):
        raw_data = pd.DataFrame({'Message': ['<p>Hello</p>', 'TEST  text'], 'Spam/Ham': [1, 0]})
        processed = preprocess_data(raw_data)
        self.assertEqual(processed.loc[0, 'Message'], 'hello')
        self.assertEqual(processed.loc[1, 'Message'], 'test text')

    def test_split_data(self):
        df = pd.DataFrame({'Message': ['a', 'b', 'c', 'd'], 'Spam/Ham': [0, 1, 0, 1]})
        train, val, test = split_data(df, test_size=0.25, random_state=42)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)

    def test_save_splits(self):
        df = pd.DataFrame({'Message': ['a', 'b', 'c', 'd'], 'Spam/Ham': [0, 1, 0, 1]})
        train, val, test = split_data(df, test_size=0.2, random_state=42)
        save_splits(train, val, test, self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'train.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'val.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test.csv')))


if __name__ == "__main__":
    unittest.main()
