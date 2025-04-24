from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import cloudpickle
from pathlib import Path

app = FastAPI()


class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text for prediction")

    @validator("text")
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Input text must be a non-empty string.")
        return value


class PredictResponse(BaseModel):
    prediction: str


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "inference_class.pkl"

with open(MODEL_PATH, "rb") as file:
    Inference = cloudpickle.load(file)

print(Inference.__doc__)

inference = Inference(model_path=str(ARTIFACTS_DIR))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    input_text = request.text
    if not input_text.strip():  # Explicitly check for empty input
        raise HTTPException(status_code=400, detail="Input text must be a non-empty string.")
    prediction = inference.predict(input_text)
    return PredictResponse(prediction=prediction)