import pytest
from fastapi.testclient import TestClient
from app import app, inference

client = TestClient(app)

def test_valid_input():
    """
    Test that the endpoint returns a valid response for a valid input.
    """
    response = client.post(
        "/predict",
        json={"text": "What a great MLOps lecture, I am very satisfied"}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], str)

def test_empty_input():
    """
    Test that the endpoint returns a 400 error for an empty input.
    """
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Input text must be a non-empty string."

def test_invalid_input():
    """
    Test that the endpoint returns a 422 error for invalid input (e.g., missing 'text' key).
    """
    response = client.post(
        "/predict",
        json={"invalid_key": "This is invalid"}
    )
    assert response.status_code == 422

def test_model_loading():
    """
    Test that the model is loaded from the cloudpickle file without errors.
    """
    assert inference is not None

def test_model_inference():
    """
    Test that the model can make predictions for a few sample strings.
    """
    sample_texts = [
        "I love this product!",
        "This is the worst experience I've ever had.",
        "It's okay, not great but not terrible."
    ]
    for text in sample_texts:
        prediction = inference.predict(text)
        assert isinstance(prediction, str)

def test_response_validation():
    """
    Test that the output is a valid JSON response.
    """
    response = client.post(
        "/predict",
        json={"text": "This is a test input"}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert isinstance(json_response["prediction"], str)