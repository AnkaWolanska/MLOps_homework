import pytest
from fastapi.testclient import TestClient
from app import app, inference

client = TestClient(app)

def test_valid_input():
    response = client.post(
        "/predict",
        json={"text": "What a great MLOps lecture, I am very satisfied"}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], str)

def test_empty_input():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 422


def test_model_loading():
    assert inference is not None

testdata = [
    ("I love this product!", "positive"),
    ("This is the worst experience I've ever had.", "negative"),
    ("It's okay, not great but not terrible.", "positive"),
]

@pytest.mark.parametrize("text, expected_prediction", testdata)
def test_model_inference(text, expected_prediction):
    prediction = inference.predict(text)
    assert prediction == expected_prediction

def test_response_validation():
    response = client.post(
        "/predict",
        json={"text": "This is a test input"}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert isinstance(json_response["prediction"], str)