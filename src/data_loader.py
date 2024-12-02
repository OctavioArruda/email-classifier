from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.tokenizer import TextTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path, model_type='naive_bayes'):
        self.file_path = file_path
        self.model_type = model_type
        self.preprocessor = TextPreprocessor(model_type=model_type)
        self.tokenizer = TextTokenizer(model_type=model_type)

    def load_data(self):
        # Load the dataset
        data = pd.read_csv(self.file_path)
        data['text'] = data['text'].apply(self.preprocessor.cleaner.clean_text)
        data['tokens'] = data['text'].apply(self.tokenizer.tokenize)
        return data

    def split_data(self, data):
        # Split into train, validation, and test sets
        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        return train_data, val_data, test_data
