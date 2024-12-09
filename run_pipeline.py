import os
import logging
from src.utils.prepare_data import extract_dataset, load_data, preprocess_data, split_data, save_splits
from src.models.naive_bayes_model import NaiveBayesModel
from src.models.bert_model import BERTModel
from src.utils.evaluate_models import evaluate_model
from src.utils.generate_summary import generate_summary
from src.utils.visualize_metrics import visualize_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Paths
    zip_path = "data/raw/enron_spam_data.zip"
    extract_to = "data/extracted"
    processed_dir = "data/processed"
    models_dir = "models"
    evaluation_dir = "evaluation_results"
    summary_file = os.path.join(evaluation_dir, "summary.txt")

    try:
        # Step 1: Prepare the dataset
        logging.info("Preparing dataset...")
        csv_path = extract_dataset(zip_path, extract_to)
        df = load_data(csv_path)
        df = preprocess_data(df)
        train, val, test = split_data(df, test_size=0.2, random_state=42)
        save_splits(train, val, test, processed_dir)

        # Step 2: Train Naive Bayes model
        logging.info("Training Naive Bayes model...")
        nb_model = NaiveBayesModel()
        nb_model.train(train['Message'], train['Spam/Ham'])
        os.makedirs(models_dir, exist_ok=True)
        nb_model.save(os.path.join(models_dir, "naive_bayes_model.pkl"))

        # Step 3: Train BERT model
        logging.info("Training BERT model...")
        bert_model = BERTModel()
        bert_model.train(train, val)
        bert_model.save(os.path.join(models_dir, "bert_model"))

        # Step 4: Evaluate models
        logging.info("Evaluating models...")
        os.makedirs(evaluation_dir, exist_ok=True)

        nb_model.load(os.path.join(models_dir, "naive_bayes_model.pkl"))
        nb_metrics = evaluate_model(nb_model, test, evaluation_dir)

        bert_model.load(os.path.join(models_dir, "bert_model"))
        bert_metrics = evaluate_model(bert_model, test, evaluation_dir)

        # Step 5: Generate summary
        logging.info("Generating summary...")
        generate_summary(nb_metrics, bert_metrics, summary_file)

        # Step 6: Visualize metrics
        logging.info("Visualizing metrics...")
        visualize_metrics(nb_metrics, bert_metrics, evaluation_dir)

        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
