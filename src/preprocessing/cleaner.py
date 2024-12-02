import re
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def get_wordnet_pos(word):
    """Map NLTK POS tag to WordNet POS tag."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,  # Adjective
        'N': wordnet.NOUN,  # Noun
        'V': wordnet.VERB,  # Verb
        'R': wordnet.ADV   # Adverb
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if no match


class TextCleaner:
    def __init__(self, remove_stopwords=True, language='english', lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.language = language
        self.lemmatize = lemmatize
        if self.remove_stopwords:
            self.stop_words = set(nltk.corpus.stopwords.words(language))
        else:
            self.stop_words = None
        self.lemmatizer = nltk.WordNetLemmatizer()

    def clean_text(self, text):
        """
        Clean and preprocess the input text.
        Steps:
        - Remove HTML tags
        - Remove URLs
        - Remove special characters and numbers
        - Convert to lowercase
        - Tokenize
        - Optionally remove stopwords
        - Lemmatize tokens with POS tagging
        """
        text = re.sub(r'<.*?>', '', text)  # Remove HTML
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.lower().strip()  # Convert to lowercase and strip spaces

        tokens = word_tokenize(text)  # Tokenize the text

        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]

        return ' '.join(tokens)
