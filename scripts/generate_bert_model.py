import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch


class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        """
        Custom dataset for BERT training.
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns a single sample (encoding and label) for the given index.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def prepare_data(file_path):
    """
    Load and preprocess data for BERT training.
    """
    data = pd.read_csv(file_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data["text"].tolist(), data["label"].tolist(), test_size=0.2
    )
    return train_texts, val_texts, train_labels, val_labels


def tokenize_data(tokenizer, texts, labels):
    """
    Tokenize the input texts and return a PyTorch dataset.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return BERTDataset(encodings, labels)


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    Save the trained BERT model and tokenizer.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


def main():
    # Configuration
    data_path = os.getenv("DATA_PATH", "data/dataset.csv")  # Path to the dataset
    model_dir = os.getenv("MODEL_DIR", "models/bert")       # Directory to save the model

    # Load and prepare data
    train_texts, val_texts, train_labels, val_labels = prepare_data(data_path)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Tokenize data
    train_dataset = tokenize_data(tokenizer, train_texts, train_labels)
    val_dataset = tokenize_data(tokenizer, val_texts, val_labels)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer, model_dir)


if __name__ == "__main__":
    main()
