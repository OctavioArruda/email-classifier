import unittest
from unittest.mock import MagicMock
from src.preprocessing.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        # Mock the cleaner and tokenizer behavior
        self.cleaner_mock = MagicMock()
        self.tokenizer_mock = MagicMock()
        
        # Patch Cleaner and TextTokenizer
        self.text_preprocessor = TextPreprocessor(model_type="naive_bayes", remove_stopwords=True)
        self.text_preprocessor.cleaner = self.cleaner_mock
        self.text_preprocessor.tokenizer = self.tokenizer_mock

    def test_preprocess_with_naive_bayes(self):
        # Mock behavior
        input_text = "This is a test email."
        cleaned_text = "test email"
        tokens = ["test", "email"]

        self.cleaner_mock.clean_text.return_value = cleaned_text
        self.tokenizer_mock.tokenize.return_value = tokens

        # Run preprocess
        result = self.text_preprocessor.preprocess(input_text)

        # Assert calls and results
        self.cleaner_mock.clean_text.assert_called_once_with(input_text)
        self.tokenizer_mock.tokenize.assert_called_once_with(cleaned_text)
        self.assertEqual(result, tokens)

    def test_preprocess_with_bert(self):
        # Switch to BERT tokenizer
        self.text_preprocessor = TextPreprocessor(model_type="bert", remove_stopwords=True)
        self.text_preprocessor.cleaner = self.cleaner_mock
        self.text_preprocessor.tokenizer = self.tokenizer_mock

        # Mock behavior
        input_text = "This is a test email."
        cleaned_text = "test email"
        tokens = [101, 7592, 2023, 102]

        self.cleaner_mock.clean_text.return_value = cleaned_text
        self.tokenizer_mock.tokenize.return_value = tokens

        # Run preprocess
        result = self.text_preprocessor.preprocess(input_text)

        # Assert calls and results
        self.cleaner_mock.clean_text.assert_called_once_with(input_text)
        self.tokenizer_mock.tokenize.assert_called_once_with(cleaned_text)
        self.assertEqual(result, tokens)

    def test_preprocess_empty_string(self):
        # Mock behavior for empty input
        input_text = ""
        cleaned_text = ""
        tokens = []

        self.cleaner_mock.clean_text.return_value = cleaned_text
        self.tokenizer_mock.tokenize.return_value = tokens

        # Run preprocess
        result = self.text_preprocessor.preprocess(input_text)

        # Assert calls and results
        self.cleaner_mock.clean_text.assert_called_once_with(input_text)
        self.tokenizer_mock.tokenize.assert_called_once_with(cleaned_text)
        self.assertEqual(result, tokens)

if __name__ == "__main__":
    unittest.main()
