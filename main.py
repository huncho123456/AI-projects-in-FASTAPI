from fastapi import FastAPI
from model.image_classifier_model import PromptModel
from service.image_classifier_service import generate_response
app = FastAPI()

BASE_URL = "/api"

@app.post(f"{BASE_URL}/predict")
def image_predictor(model: PromptModel):
    return generate_response(model.image_path, model.prompt)


