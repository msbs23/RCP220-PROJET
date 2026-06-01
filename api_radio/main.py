from fastapi import FastAPI, UploadFile, File
from api_radio.schemas import PredictionResponse
from api_radio.singleton_model import ModelSingleton
from api_radio.preprocessing import preprocess_image
import torch

app = FastAPI(
    title="API Détection Anomalies Radiologiques",
    description="Sytème d'aide à la décision du radiologue - RCP220",
    version="1.0",
)

@app.on_event("startup")
async def startup():
    ModelSingleton.get_instance()

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    tensor = preprocess_image(image_bytes=image_bytes)
    feature_extractor, head = ModelSingleton.get_instance()

    with torch.no_grad():
        features = feature_extractor(tensor)
        output = head(features)
        probabilite = torch.sigmoid(output).item()

    label = "malade" if probabilite > 0.5 else "sain"

    return PredictionResponse(
        label=label,
        probabilite=round(probabilite,4),
    )
