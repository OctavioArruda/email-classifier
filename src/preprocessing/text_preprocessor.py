from src.preprocessing.cleaner import TextPreprocessor as Cleaner
from src.preprocessing.tokenizer import TextTokenizer
from nltk.corpus import stopwords
import string


class TextPreprocessor:
    def __init__(self, model_type='naive_bayes', remove_stopwords=True, language='english', lemmatize=False):
        """
        Combines text cleaning and tokenization for preprocessing.
        :param model_type: 'naive_bayes' or 'bert' for tokenizer type.
        :param remove_stopwords: Whether to remove stopwords during cleaning.
        :param language: Language for stopwords (default is English).
        :param lemmatize: Whether to apply lemmatization to text.
        """
        self.cleaner = Cleaner(remove_stopwords=remove_stopwords, language=language, lemmatize=lemmatize)
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


class Cleaner:
    def __init__(self, remove_stopwords=True, language='english', lemmatize=False):
        """
        Cleans input text by removing stopwords, punctuation, etc.
        :param remove_stopwords: Whether to remove stopwords.
        :param language: Language for stopwords.
        :param lemmatize: Whether to apply lemmatization.
        """
        self.remove_stopwords = remove_stopwords
        self.language = language
        self.lemmatize = lemmatize
        if remove_stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = None

    def clean_text(self, text):
        """
        Clean text by lowercasing, removing punctuation, and optional stopword removal.
        :param text: The input text.
        :return: Cleaned text.
        """
        text = text.lower()  # Lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        words = text.split()

        if self.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        if self.lemmatize:
            # Add lemmatization logic here (e.g., using nltk.WordNetLemmatizer or spaCy)
            pass

        return ' '.join(words)
