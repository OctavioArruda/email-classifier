import unittest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestRunPipeline(unittest.TestCase):
    @patch("src.utils.prepare_data.extract_dataset")
    @patch("src.utils.prepare_data.load_data")
    @patch("src.utils.prepare_data.preprocess_data")
    @patch("src.utils.prepare_data.split_data")
    @patch("src.utils.prepare_data.save_splits")
    @patch("src.models.naive_bayes_model.NaiveBayesModel")
    @patch("src.models.bert_model.BERTModel")
    @patch("src.utils.evaluate_models.evaluate_model")
    def test_run_pipeline(self, mock_evaluate, mock_bert, mock_nb, mock_save_splits,
                          mock_split, mock_preprocess, mock_load, mock_extract):
        # Mock dataset and models
        mock_extract.return_value = "mock_csv_path"
        mock_load.return_value = pd.DataFrame({
            'Message ID': [1, 2],
            'Message': ['Spam message', 'Ham message'],
            'Spam/Ham': [1, 0]
        })
        mock_preprocess.return_value = pd.DataFrame({
            'Message ID': [1, 2],
            'Message': ['spam message', 'ham message'],
            'Spam/Ham': [1, 0]
        })
        mock_split.return_value = (
            pd.DataFrame({'Message': ['spam message'], 'Spam/Ham': [1]}),  # train
            pd.DataFrame({'Message': ['ham message'], 'Spam/Ham': [0]}),  # val
            pd.DataFrame({'Message': ['spam message', 'ham message'], 'Spam/Ham': [1, 0]})  # test
        )

        # Mock models
        mock_nb_instance = mock_nb.return_value
        mock_bert_instance = mock_bert.return_value

        # Run pipeline
        from run_pipeline import main
        main()

        # Assertions
        mock_extract.assert_called_once()
        mock_load.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_split.assert_called_once()
        mock_save_splits.assert_called_once()
        mock_nb_instance.train.assert_called_once()
        mock_bert_instance.train.assert_called_once()
        mock_nb_instance.save.assert_called_once()
        mock_bert_instance.save.assert_called_once()
        mock_evaluate.assert_called()

        print("Pipeline test completed successfully.")


if __name__ == "__main__":
    unittest.main()
