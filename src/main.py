from src.data_loader import DataLoader
from models.naive_bayes_model import NaiveBayesModel
from models.bert_model import BERTModel

def main():
    # Load data
    data_loader = DataLoader(file_path='path_to_enron_spam_dataset.csv')
    data = data_loader.load_data()
    train_data, val_data, test_data = data_loader.split_data(data)

    # Train Naive Bayes model
    nb_model = NaiveBayesModel()
    nb_model.train(train_data)
    nb_accuracy, nb_report = nb_model.evaluate(test_data)
    print("Naive Bayes Accuracy:", nb_accuracy)
    print("Naive Bayes Report:\n", nb_report)

    # Train BERT model
    bert_model = BERTModel()
    bert_model.train(
        train_data['text'], train_data['label'],
        val_data['text'], val_data['label']
    )
    bert_results = bert_model.evaluate(test_data['text'], test_data['label'])
    print("BERT Evaluation Results:", bert_results)

if __name__ == "__main__":
    main()
