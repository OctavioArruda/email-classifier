import unittest  # Import the unittest module
from src.preprocessing.text_preprocessor import TextPreprocessor  # Ensure this path is correct

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)

    def test_clean_text_handles_empty_string(self):
        input_text = ""
        expected_output = ""
        self.assertEqual(self.preprocessor.cleaner.clean_text(input_text), expected_output)

    def test_clean_text_handles_numbers(self):
        input_text = "The total cost is 123 dollars"
        expected_output = "total cost dollar"
        self.assertEqual(self.preprocessor.cleaner.clean_text(input_text), expected_output)

    def test_clean_text_handles_punctuation_and_numbers(self):
        input_text = "Well, it's 100% true!"
        expected_output = "well true"
        self.assertEqual(self.preprocessor.cleaner.clean_text(input_text), expected_output)
