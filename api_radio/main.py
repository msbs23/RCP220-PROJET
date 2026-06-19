import time
from datetime import timedelta

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from api_radio.schemas import PredictionResponse, Token
from api_radio.auth import authenticate_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from api_radio.singleton_model import ModelSingleton
from api_radio.preprocessing import preprocess_image
from prometheus_fastapi_instrumentator import Instrumentator
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

Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def startup():
    ModelSingleton.get_instance()
    logger.info("Modèle chargé et API prête")


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer")


@app.post("/feedback")
async def feedback(prediction_id: str, correct: bool, _: dict = Depends(get_current_user)):
    logger.info(f"FEEDBACK | correct={correct} | id={prediction_id}")
    return {"status": "recorded"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...), _: dict = Depends(get_current_user)):
    start = time.time()

    image_bytes = await image.read()
    tensor = preprocess_image(image_bytes=image_bytes)
    feature_extractor, head = ModelSingleton.get_instance()

    with torch.no_grad():
        features = feature_extractor(tensor)
        output = head(features)
        probabilite = torch.sigmoid(output).item()

    label = "malade" if probabilite > 0.5 else "sain"
    latence = round(time.time() - start, 4)

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
        probabilite=round(probabilite, 4),
    )