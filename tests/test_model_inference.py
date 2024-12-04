import unittest
from unittest.mock import MagicMock, patch
from src.inference.model_inference import ModelInference
import torch


class TestModelInference(unittest.TestCase):
    def setUp(self):
        # Patch BERTModel and NaiveBayesModel
        self.bert_model_mock = patch("src.models.bert_model.BERTModel").start()
        self.naive_bayes_model_mock = patch("src.models.naive_bayes_model.NaiveBayesModel").start()

        # Mock BERTModel and NaiveBayesModel behavior
        self.bert_model_mock.return_value.predict = MagicMock(return_value="ham")
        self.bert_model_mock.return_value.load = MagicMock()
        self.naive_bayes_model_mock.return_value.predict = MagicMock(return_value="spam")
        self.naive_bayes_model_mock.return_value.load = MagicMock()

    def tearDown(self):
        patch.stopall()

    def test_bert_model_inference(self):
        inference = ModelInference(model_type="bert", model_dir="invalid_dir/")
        prediction = inference.predict("This is a test email.")
        self.assertEqual(prediction, "mocked_label")

    def test_naive_bayes_model_inference(self):
        inference = ModelInference(model_type="naive_bayes", model_dir="invalid_dir/")
        prediction = inference.predict("This is a spam email.")
        self.assertEqual(prediction, "mocked_label")

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError) as context:
            ModelInference(model_type="unsupported_model", model_dir="models/")
        self.assertIn("Unsupported model type", str(context.exception))

    def test_model_directory_not_found(self):
        inference = ModelInference(model_type="bert", model_dir="invalid_dir/")
        self.assertIsNotNone(inference.model)
        self.assertEqual(inference.predict("test input"), "mocked_label")

    def test_predict_invalid_input(self):
        inference = ModelInference(model_type="bert", model_dir="invalid_dir/")
        with self.assertRaises(ValueError) as context:
            inference.predict(12345)  # Invalid input
        self.assertIn("Input text must be a string", str(context.exception))

    def test_empty_text(self):
        inference = ModelInference(model_type="bert", model_dir="invalid_dir/")
        prediction = inference.predict("")
        self.assertEqual(prediction, "mocked_label")

    def test_large_text_input(self):
        long_text = "This is a " + "very " * 1000 + "long email."
        inference = ModelInference(model_type="bert", model_dir="invalid_dir/")
        prediction = inference.predict(long_text)
        self.assertEqual(prediction, "mocked_label")


if __name__ == "__main__":
    unittest.main()
