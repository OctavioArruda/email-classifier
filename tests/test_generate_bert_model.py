import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from scripts.generate_bert_model import main, save_model_and_tokenizer


@patch("pandas.read_csv")
@patch("scripts.generate_bert_model.Trainer.train")
@patch("scripts.generate_bert_model.save_model_and_tokenizer")
def test_main(mock_save_model, mock_train, mock_read_csv):
    """
    Test the main function of generate_bert_model.py.
    """
    # Mock the dataset
    mock_read_csv.return_value = pd.DataFrame({
        "text": ["Sample text 1", "Sample text 2"],
        "label": [0, 1]
    })

    # Mock training process and save process
    mock_train.return_value = None
    mock_save_model.return_value = None

    # Mock the environment variables
    os.environ["DATA_PATH"] = "data/mock_dataset.csv"
    os.environ["MODEL_DIR"] = "mock_model_dir"

    # Call the main function
    main()

    # Assert that train and save_model_and_tokenizer were called
    mock_train.assert_called_once()
    mock_save_model.assert_called_once()


def test_save_model_and_tokenizer(tmp_path):
    """
    Test save_model_and_tokenizer function.
    """
    # Create a mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Call the save function
    output_dir = tmp_path / "bert"
    save_model_and_tokenizer(mock_model, mock_tokenizer, output_dir)

    # Assert that save_pretrained was called on both model and tokenizer
    mock_model.save_pretrained.assert_called_once_with(output_dir)
    mock_tokenizer.save_pretrained.assert_called_once_with(output_dir)
