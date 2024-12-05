import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pytest

@pytest.fixture
def setup_model_directory():
    # Ensure the models directory is empty before the test
    model_dir = "models"
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, file))
    os.makedirs(model_dir, exist_ok=True)
    yield model_dir
    # Clean up after the test
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))

def test_generate_model_and_vectorizer(setup_model_directory):
    # Simulate the script logic
    train_data = {
        "text": ["This is spam", "This is ham", "Spam messages are annoying", "Ham is delicious"],
        "label": [1, 0, 1, 0]
    }
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data["text"])
    model = MultinomialNB()
    model.fit(X_train, train_data["label"])

    # Save the model and vectorizer
    model_dir = setup_model_directory
    with open(os.path.join(model_dir, "naive_bayes_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    # Assertions
    assert os.path.exists(os.path.join(model_dir, "naive_bayes_model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "vectorizer.pkl"))

    # Load and validate
    with open(os.path.join(model_dir, "naive_bayes_model.pkl"), "rb") as f:
        loaded_model = pickle.load(f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
        loaded_vectorizer = pickle.load(f)

    # Ensure the loaded model and vectorizer work correctly
    test_text = ["Spam email is annoying"]
    X_test = loaded_vectorizer.transform(test_text)
    prediction = loaded_model.predict(X_test)
    assert prediction[0] == 1  # Ensure prediction matches expected "spam" label
