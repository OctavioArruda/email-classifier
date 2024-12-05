from fastapi.testclient import TestClient
from src.api.api_handler import app

# Initialize the test client
client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Email Classifier API is running!"}

def test_predict_valid_email():
    payload = {"text": "This is a spam email about winning money"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "classification" in response.json()
    assert response.json()["classification"] == "spam"

def test_predict_missing_text():
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # FastAPI validation error

def test_predict_with_unloaded_model():
    """
    Test the predict endpoint when the model is not loaded.
    """
    # Unload the model instance used in the API
    from src.api.api_handler import model  # Ensure this is the actual model instance
    model.unload()  # Unload the model properly

    payload = {"text": "Test email"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Model is not loaded"
