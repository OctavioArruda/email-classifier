import os
from src.utils.prepare_data import extract_dataset, load_data, preprocess_data, split_data, save_splits
from src.models.naive_bayes_model import NaiveBayesModel
from src.models.bert_model import BERTModel
from src.utils.evaluate_models import evaluate_model


def main():
    # Paths
    zip_path = "data/raw/enron_spam_data.zip"
    extract_to = "data/extracted"
    processed_dir = "data/processed"
    models_dir = "models"
    evaluation_dir = "evaluation_results"

    # Step 1: Prepare the dataset
    print("Preparing dataset...")
    csv_path = extract_dataset(zip_path, extract_to)
    df = load_data(csv_path)
    df = preprocess_data(df)
    train, val, test = split_data(df, test_size=0.2, random_state=42)
    save_splits(train, val, test, processed_dir)

    # Step 2: Train Naive Bayes model
    print("Training Naive Bayes model...")
    nb_model = NaiveBayesModel()
    nb_model.train(train['Message'], train['Spam/Ham'])
    os.makedirs(models_dir, exist_ok=True)
    nb_model.save(os.path.join(models_dir, "naive_bayes_model.pkl"))

    # Step 3: Train BERT model
    print("Training BERT model...")
    bert_model = BERTModel()
    bert_model.train(train, val)
    bert_model.save(os.path.join(models_dir, "bert_model"))

    # Step 4: Evaluate models
    print("Evaluating models...")
    test_data = test  # Directly using test split from Step 1

    # Evaluate Naive Bayes
    nb_model.load(os.path.join(models_dir, "naive_bayes_model.pkl"))
    evaluate_model(nb_model, test_data, evaluation_dir)

    # Evaluate BERT
    bert_model.load(os.path.join(models_dir, "bert_model"))
    evaluate_model(bert_model, test_data, evaluation_dir)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
