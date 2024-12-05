import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.naive_bayes_model import NaiveBayesModel
from src.utils.encoders import custom_jsonable_encoder

# Initialize FastAPI app
app = FastAPI()

# Load the Naive Bayes model
model = NaiveBayesModel()
model_dir = os.getenv("MODEL_DIR", "models")
try:
    model.load(model_dir)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Input schema for the API
class EmailText(BaseModel):
    text: str

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {"message": "Email Classifier API is running!"}

@app.post("/predict")
async def predict(email: EmailText):
    """
    Predict endpoint that uses the custom encoder.
    """
    # Explicitly check if the model is loaded
    if not model.is_loaded():
        raise HTTPException(status_code=400, detail="Model is not loaded")

    # Simulating a model prediction with a numpy type
    prediction = {
        "text": email.text,
        "prediction": np.int64(1),  # Simulated output
        "confidence": np.float64(0.95),
        "classification": "spam" if np.int64(1) == 1 else "not spam"  # Add human-readable label
    }
    # Encode the response using the custom encoder
    encoded_response = custom_jsonable_encoder(prediction)
    return encoded_response


