from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    input_text = request.text
    if "great" in input_text.lower():
        prediction = "positive"
    else:
        prediction = "negative"

    return PredictResponse(prediction=prediction)
