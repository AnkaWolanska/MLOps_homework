from fastapi import FastAPI
from pydantic import BaseModel
import cloudpickle
from pathlib import Path
from artifacts.inference import Inference

app = FastAPI()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "inference_class.pkl"

# Using cloudpickle causes seg fault during deserializaton process
# with open(MODEL_PATH, "rb") as file:
#     Inference = cloudpickle.load(file)

print(Inference.__doc__)

inference = Inference(model_path=str(ARTIFACTS_DIR))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    input_text = request.text
    prediction = inference.predict(input_text)
    return PredictResponse(prediction=prediction)
