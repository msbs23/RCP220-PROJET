import torch
import torch.nn as nn
from torchvision import models
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")


def _build_model(weights_path: str):
    feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.fc = nn.Identity()

    head = nn.Linear(2048, 1)
    head.load_state_dict(torch.load(weights_path, map_location="cpu"))

    feature_extractor.eval()
    head.eval()
    return feature_extractor, head


def load_model():
    if MODEL_PATH:
        print(f"Chargement du modèle depuis : {MODEL_PATH}")
        return _build_model(MODEL_PATH)

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_auc DESC", "attributes.start_time DESC"]
    )

    best_run = None
    for run in runs:
        if client.list_artifacts(run.info.run_id):
            best_run = run
            break

    if best_run is None:
        raise Exception("Aucun run MLflow avec artifacts trouvé !")

    weights_path = client.download_artifacts(
        best_run.info.run_id,
        "head_semi_supervised_v1.0_auc0718.pth"
    )
    print(f"Modèle chargé via MLflow : run {best_run.info.run_id}, AUC={best_run.data.metrics['val_auc']}")
    return _build_model(weights_path)