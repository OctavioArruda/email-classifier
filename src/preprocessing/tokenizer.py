from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK resources
nltk.download('punkt')

class TextTokenizer:
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the tokenizer based on the model type.
        :param model_type: 'naive_bayes' or 'bert'
        """
        self.model_type = model_type
        if model_type == 'bert':
            # Initialize the BERT tokenizer from Hugging Face
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_type == 'naive_bayes':
            # Use NLTK's word tokenizer for Naive Bayes
            self.tokenizer = word_tokenize
        else:
            raise ValueError("Unsupported model_type. Choose 'naive_bayes' or 'bert'.")

    def tokenize(self, text):
        """
        Tokenize text based on the model type.
        :param text: The input text string.
        :return: List of tokens or token IDs.
        """
        if self.model_type == 'bert':
            # Tokenize using Hugging Face BERT tokenizer and include special tokens
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
        elif self.model_type == 'naive_bayes':
            # Tokenize using NLTK
            tokens = self.tokenizer(text)
        else:
            raise ValueError("Unsupported model_type. Choose 'naive_bayes' or 'bert'.")

        return tokens
