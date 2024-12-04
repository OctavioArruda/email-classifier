import unittest
from unittest.mock import MagicMock, patch
from src.models.model_trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "This is a spam email.",
            "This is a regular email.",
            "Get rich quick!",
            "Meeting scheduled for tomorrow.",
        ]
        self.labels = [1, 0, 1, 0]

    @patch("src.models.model_trainer.BERTModel")
    def test_train_with_bert(self, MockBERTModel):
        mock_bert = MockBERTModel.return_value
        trainer = ModelTrainer(model_type="bert")

        trainer.train(self.texts, self.labels)

        # Ensure BERT's train method is called
        mock_bert.train.assert_called_once()

    @patch("src.models.model_trainer.NaiveBayesModel")
    def test_train_with_naive_bayes(self, MockNaiveBayesModel):
        mock_nb = MockNaiveBayesModel.return_value
        trainer = ModelTrainer(model_type="naive_bayes")

        trainer.train(self.texts, self.labels)

        # Ensure Naive Bayes's train method is called
        mock_nb.train.assert_called_once()

    @patch("src.models.model_trainer.BERTModel")
    def test_evaluate_with_bert(self, MockBERTModel):
        mock_bert = MockBERTModel.return_value
        mock_bert.evaluate.return_value = (0.9, "Mock Report")
        trainer = ModelTrainer(model_type="bert")

        accuracy, report = trainer.evaluate(self.texts, self.labels)

        # Ensure evaluate is called and returns the mocked values
        mock_bert.evaluate.assert_called_once()
        self.assertEqual(accuracy, 0.9)
        self.assertEqual(report, "Mock Report")

    @patch("src.models.model_trainer.NaiveBayesModel")
    def test_evaluate_with_naive_bayes(self, MockNaiveBayesModel):
        mock_nb = MockNaiveBayesModel.return_value
        mock_nb.evaluate.return_value = (0.85, "Mock NB Report")
        trainer = ModelTrainer(model_type="naive_bayes")

        accuracy, report = trainer.evaluate(self.texts, self.labels)

        # Ensure evaluate is called and returns the mocked values
        mock_nb.evaluate.assert_called_once()
        self.assertEqual(accuracy, 0.85)
        self.assertEqual(report, "Mock NB Report")

    @patch("os.makedirs")
    @patch("src.models.model_trainer.BERTModel")
    def test_save_model(self, MockBERTModel, mock_makedirs):
        mock_bert = MockBERTModel.return_value
        trainer = ModelTrainer(model_type="bert")

        trainer.save_model(output_dir="mock_dir")

        # Ensure save is called
        mock_bert.save.assert_called_once_with("mock_dir")
        mock_makedirs.assert_called_once_with("mock_dir", exist_ok=True)

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError) as context:
            ModelTrainer(model_type="invalid_model")
        self.assertEqual(
            str(context.exception), "Unsupported model type: invalid_model"
        )


if __name__ == "__main__":
    unittest.main()
