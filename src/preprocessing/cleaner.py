import re
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def get_wordnet_pos(word):
    """Map NLTK POS tag to WordNet POS tag."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,    # Adjective
        'N': wordnet.NOUN,   # Noun
        'V': wordnet.VERB,   # Verb
        'R': wordnet.ADV     # Adverb
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if no match

class TextPreprocessor:
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
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
        # Clean and tokenize
        text = re.sub(r'<.*?>', '', text)  # Remove HTML
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        tokens = nltk.word_tokenize(text)

        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
            tokens = [
                self.lemmatizer.lemmatize(word, get_wordnet_pos(word))
                if get_wordnet_pos(word) != wordnet.ADV
                else word.rstrip('ly') if word.endswith('ly') else word
                for word in tokens
            ]

        return ' '.join(tokens)
