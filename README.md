# Email Classifier: A Machine Learning-Based Spam Detection System

## About
The **Email Classifier** is a machine learning-based project that classifies emails as **spam** or **ham** using advanced Natural Language Processing (NLP) techniques. This repository implements two models for classification:
- **Naive Bayes Classifier**: A lightweight, interpretable baseline model.
- **BERT Transformer Model**: A powerful deep learning-based model leveraging pre-trained transformers for state-of-the-art results.

This project is designed to explore the trade-offs between traditional machine learning algorithms and deep learning approaches in real-world spam detection scenarios.

---

## Features
- **Preprocessing Pipelines**:
  - Tokenization, lemmatization, stopword removal, and feature extraction for classical models.
  - BERT tokenizer integration for deep learning.
- **Model Training and Evaluation**:
  - Naive Bayes Classifier with `sklearn`.
  - BERT fine-tuning with `transformers`.
- **Comprehensive Metrics**:
  - Accuracy, Precision, Recall, and F1-Score.
- **Unit Testing**:
  - Full test coverage using `unittest` for both models.
- **Integration with Hugging Face Transformers**:
  - Implements modern NLP techniques with ease.

---

## Technologies Used
- **Python 3.11**
- **Libraries**:
  - NLP: `nltk`, `transformers`
  - Machine Learning: `sklearn`, `torch`
  - Testing: `unittest`, `unittest.mock`
- **Development Tools**:
  - `pip`, `pytest`, `jupyter`

---

## Repository Structure
```
email-classifier/
├── src/
│   ├── models/
│   │   ├── bert_model.py       # BERT-based spam classifier
│   │   ├── naive_bayes_model.py # Naive Bayes spam classifier
│   ├── preprocessing/
│   │   ├── text_preprocessor.py # Text preprocessing utilities
│   └── utils/
│       ├── helper_functions.py  # Reusable functions
├── tests/
│   ├── test_bert_model.py       # Unit tests for BERT model
│   ├── test_naive_bayes_model.py # Unit tests for Naive Bayes model
│   └── test_text_preprocessor.py # Unit tests for preprocessing
├── data/
│   ├── raw/                     # Raw email datasets
│   ├── processed/               # Preprocessed datasets
├── README.md                    # Project overview
└── requirements.txt             # Dependency list
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/email-classifier.git
   cd email-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLP data:
   ```bash
   python -m nltk.downloader all
   ```

---

## Usage
1. **Train and evaluate the Naive Bayes Model**:
   Run the following command to train and evaluate the Naive Bayes classifier:
   ```bash
   python -m src.models.naive_bayes_model
   ```

2. **Train and evaluate the BERT Model**:
   Run the following command to train and evaluate the BERT-based classifier:
   ```bash
   python -m src.models.bert_model
   ```

---

## Contributing
Contributions are welcome! Please follow these steps:
1. **Fork the repository**:
   Click on the "Fork" button at the top-right of this repository.
2. **Create a new feature branch**:
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Commit your changes**:
   Add your changes and commit:
   ```bash
   git commit -m "Add your feature"
   ```
4. **Push to your forked repository**:
   ```bash
   git push origin feature/your-feature
   ```
5. **Open a Pull Request**:
   Navigate to your forked repository on GitHub and click on "New Pull Request."

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
Special thanks to the developers of:
- [Hugging Face Transformers](https://huggingface.co/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
