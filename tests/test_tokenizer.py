import unittest
from src.preprocessing.tokenizer import TextTokenizer

class TestTextTokenizer(unittest.TestCase):
    def setUp(self):
        # Create instances for both model types
        self.naive_tokenizer = TextTokenizer(model_type='naive_bayes')
        self.bert_tokenizer = TextTokenizer(model_type='bert')

    def test_naive_bayes_tokenizer(self):
        input_text = "This is a test email."
        expected_tokens = ['This', 'is', 'a', 'test', 'email', '.']
        actual_tokens = self.naive_tokenizer.tokenize(input_text)
        self.assertEqual(actual_tokens, expected_tokens, "Naive Bayes tokenizer failed.")

    def test_naive_bayes_empty_string(self):
        input_text = ""
        expected_tokens = []
        actual_tokens = self.naive_tokenizer.tokenize(input_text)
        self.assertEqual(actual_tokens, expected_tokens, "Naive Bayes tokenizer failed for empty string.")

    def test_bert_tokenizer(self):
        input_text = "This is a test email."
        # Ensure output contains token IDs (list of integers) and includes special tokens
        actual_tokens = self.bert_tokenizer.tokenize(input_text)
        self.assertIsInstance(actual_tokens, list, "BERT tokenizer output should be a list.")
        self.assertGreater(len(actual_tokens), 2, "BERT tokenizer should include special tokens.")
        self.assertTrue(all(isinstance(token, int) for token in actual_tokens), "BERT tokenizer output should be integers.")

    def test_bert_empty_string(self):
        input_text = ""
        # BERT tokenizer should return a list with only special tokens [CLS] and [SEP]
        actual_tokens = self.bert_tokenizer.tokenize(input_text)
        self.assertIsInstance(actual_tokens, list, "BERT tokenizer output should be a list.")
        self.assertEqual(len(actual_tokens), 2, "BERT tokenizer should return only special tokens for empty input.")

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            _ = TextTokenizer(model_type='unsupported_model')

if __name__ == "__main__":
    unittest.main()
