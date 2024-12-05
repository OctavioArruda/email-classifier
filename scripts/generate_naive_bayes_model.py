import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
train_data = {
    "text": [
        "Win a lottery now", 
        "Buy cheap products online", 
        "Meeting rescheduled to Monday", 
        "Discounts on your favorite brands", 
        "Team lunch planned for today", 
        "Click here to claim your prize", 
        "Let's meet tomorrow for the project discussion", 
        "Exclusive deals just for you"
    ],
    "label": [1, 1, 0, 1, 0, 1, 0, 1]
}

# Initialize and train the model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data["text"])
model = MultinomialNB()
model.fit(X_train, train_data["label"])

# Save the model and vectorizer to the models/ directory
import os
os.makedirs("models", exist_ok=True)
with open("models/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved to 'models/' directory.")
