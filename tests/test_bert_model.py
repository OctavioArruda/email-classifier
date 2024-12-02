import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
from src.models.bert_model import BERTModel


class MockBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_labels": 2})
        self.linear = nn.Linear(128, self.config.num_labels)  # Simulated linear layer for logits

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        print(f"MockBertModel.forward - input_ids: {input_ids}, attention_mask: {attention_mask}, labels: {labels}")
        batch_size = input_ids.size(0)
        logits = self.linear(torch.randn(batch_size, 128))  # Simulated logits

        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


class TestBERTModel(unittest.TestCase):
    @patch("src.models.bert_model.BertTokenizer.from_pretrained")
    def setUp(self, mock_tokenizer_class):
        # Mock tokenizer
        self.mock_tokenizer = mock_tokenizer_class.return_value
        self.mock_tokenizer.side_effect = lambda text, truncation, padding, max_length, return_tensors: {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 13478, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])
        }

        # Mock model
        self.mock_model = MockBertModel()

        # Initialize the BERTModel
        self.model = BERTModel()
        self.model.model = self.mock_model

    def test_train(self):
        train_texts = ["spam email 1", "ham email 2"]
        train_labels = [1, 0]
        val_texts = ["spam email 3", "ham email 4"]
        val_labels = [1, 0]

        self.model.train(train_texts, train_labels, val_texts, val_labels)

    def test_evaluate(self):
        test_texts = ["spam email 5"]
        test_labels = [1]

        self.model.evaluate(test_texts, test_labels)


if __name__ == "__main__":
    unittest.main()
