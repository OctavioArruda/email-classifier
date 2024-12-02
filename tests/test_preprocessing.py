import unittest
from src.preprocessing.cleaner import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(remove_stopwords=True)

    def test_clean_text_handles_empty_string(self):
        input_text = ""
        expected_output = ""
        self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_clean_text_handles_numbers(self):
        input_text = "The total cost is 123 dollars"
        expected_output = "total cost dollar"
        self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_clean_text_handles_mixed_casing(self):
        input_text = "This Is A TeSt Email"
        expected_output = "test email"
        self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_clean_text_handles_repeated_spaces(self):
        input_text = "This     is   a   test"
        expected_output = "test"
        self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_clean_text_handles_special_characters_only(self):
        input_text = "@#$%^&*()!"
        expected_output = ""
        self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_clean_text_handles_stopwords_disabled(self):
        input_text = "This is a simple test"
        expected_output = "this is a simple test"
        preprocessor_no_stopwords = TextPreprocessor(remove_stopwords=False)
        self.assertEqual(preprocessor_no_stopwords.clean_text(input_text), expected_output)

    def test_clean_text_handles_lemmas(self):
        input_text = "Running runners run quickly"
        expected_output = "run runner run quick"
        self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

if __name__ == "__main__":
    unittest.main()
