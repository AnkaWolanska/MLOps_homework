from fastapi import FastAPI
from pydantic import BaseModel
import cloudpickle
from pathlib import Path

app = FastAPI()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


ARTIFACTS_DIR = Path("app/artifacts")
MODEL_PATH = ARTIFACTS_DIR / "inference_class.pkl"

with open(MODEL_PATH, "rb") as file:
    Inference = cloudpickle.load(file)

print(Inference.__doc__)

inference = Inference(...)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    input_text = request.text
    prediction = inference.predict(input_text)
    return PredictResponse(prediction=prediction)
