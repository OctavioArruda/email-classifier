import unittest
from unittest.mock import patch, MagicMock
from run_pipeline import main


class TestRunPipeline(unittest.TestCase):
    @patch("run_pipeline.visualize_metrics")  # Patch visualize_metrics
    @patch("run_pipeline.generate_summary")  # Patch generate_summary
    @patch("run_pipeline.evaluate_model")  # Patch evaluate_model
    @patch("run_pipeline.BERTModel")  # Patch BERTModel
    @patch("run_pipeline.NaiveBayesModel")  # Patch NaiveBayesModel
    @patch("run_pipeline.save_splits")  # Patch save_splits
    @patch("run_pipeline.split_data")  # Patch split_data
    @patch("run_pipeline.preprocess_data")  # Patch preprocess_data
    @patch("run_pipeline.load_data")  # Patch load_data
    @patch("run_pipeline.extract_dataset")  # Patch extract_dataset
    def test_main(
        self,
        mock_extract_dataset,
        mock_load_data,
        mock_preprocess_data,
        mock_split_data,
        mock_save_splits,
        mock_nb_model,
        mock_bert_model,
        mock_evaluate_model,
        mock_generate_summary,
        mock_visualize_metrics,
    ):
        # Mock all pipeline steps
        mock_extract_dataset.return_value = "mock_csv_path"
        mock_load_data.return_value = MagicMock()
        mock_preprocess_data.return_value = MagicMock()
        
        # Create mock DataFrames for train, val, and test
        train = MagicMock()
        train.__getitem__.side_effect = lambda key: ["sample message"] if key == "Message" else ["ham"]
        val = MagicMock()
        test = MagicMock()
        test.__getitem__.side_effect = lambda key: ["sample test message"] if key == "Message" else ["spam"]
        mock_split_data.return_value = (train, val, test)
        
        mock_nb_instance = mock_nb_model.return_value
        mock_bert_instance = mock_bert_model.return_value
        mock_evaluate_model.side_effect = [
            {"precision": 0.85, "recall": 0.9, "f1-score": 0.87},
            {"precision": 0.9, "recall": 0.95, "f1-score": 0.92},
        ]
        mock_generate_summary.return_value = None
        mock_visualize_metrics.return_value = None

        # Run the pipeline
        main()

        # Assertions
        mock_extract_dataset.assert_called_once()
        mock_load_data.assert_called_once()
        mock_preprocess_data.assert_called_once()
        mock_split_data.assert_called_once()
        mock_save_splits.assert_called_once()
        mock_nb_instance.train.assert_called_once_with(["sample message"], ["ham"])
        mock_bert_instance.train.assert_called_once_with(train, val)
        mock_nb_instance.save.assert_called_once()
        mock_bert_instance.save.assert_called_once()
        mock_evaluate_model.assert_called()
        mock_generate_summary.assert_called_once()
        mock_visualize_metrics.assert_called_once()


if __name__ == "__main__":
    unittest.main()
