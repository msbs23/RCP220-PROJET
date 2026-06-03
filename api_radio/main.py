import time

from fastapi import FastAPI, UploadFile, File
from api_radio.schemas import PredictionResponse
from api_radio.singleton_model import ModelSingleton
from api_radio.preprocessing import preprocess_image
import torch
import logging


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("api_radio")

app = FastAPI(
    title="API Détection Anomalies Radiologiques",
    description="Sytème d'aide à la décision du radiologue - RCP220",
    version="1.0",
)

@app.on_event("startup")
async def startup():
    ModelSingleton.get_instance()
    logger.info("Modèle chargé et API prête")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    start = time.time()

    image_bytes = await image.read()
    tensor = preprocess_image(image_bytes=image_bytes)
    feature_extractor, head = ModelSingleton.get_instance()

    with torch.no_grad():
        features = feature_extractor(tensor)
        output = head(features)
        probabilite = torch.sigmoid(output).item()

    label = "malade" if probabilite > 0.5 else "sain"
    latence = round(time.time() - start,4)

    # ================== KPI modèle : score, label) ==========================#
    logger.info(
        f"PREDICT | label={label} | "
        f"probabilite={probabilite:.4f} | "
        f"latence={latence} | "
        f"fichier={image.filename}"
    )

    # =================== ALerte automatique : LANTENCE + AMBIGUITÉ =================================#
    if latence > 2.0:
        logger.warning(f"ALERTE LATENCE | {latence}s > seuil 2.0s")

    if 0.4 < probabilite < 0.6:
        logger.warning(
            f"ALERTE AMBIGUÏTÉ | probabilite={round(probabilite, 4)} "
            f"— supervision humaine requise (radiologue)"
        )

    return PredictionResponse(
        label=label,
        probabilite=round(probabilite,4),
    )
