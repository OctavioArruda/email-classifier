from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.tokenizer import TextTokenizer


class TextPreprocessor:
    def __init__(self, model_type="naive_bayes", remove_stopwords=True, lemmatize=True):
        """
        Combines text cleaning and tokenization for preprocessing.
        :param model_type: 'naive_bayes' or 'bert' for tokenizer type.
        :param remove_stopwords: Whether to remove stopwords during cleaning.
        :param lemmatize: Whether to apply lemmatization to text.
        """
        self.cleaner = TextCleaner(remove_stopwords=remove_stopwords, lemmatize=lemmatize)
        self.tokenizer = TextTokenizer(model_type=model_type)

    def preprocess(self, text):
        """
        Preprocess a single text by cleaning and tokenizing it.
        :param text: The input text string.
        :return: List of tokens for naive_bayes, or token IDs for bert.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        cleaned_text = self.cleaner.clean_text(text)
        tokens = self.tokenizer.tokenize(cleaned_text)
        return tokens
