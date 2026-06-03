import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Depuis le host, MLflow est sur localhost:5000 (port mappé par docker-compose)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# mlflow:5000 est valide uniquement dans le réseau Docker interne
if "mlflow:" in tracking_uri:
    tracking_uri = "http://localhost:5000"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("rcp220_semi_supervised")

with mlflow.start_run(run_name="v1.0"):
    mlflow.log_param("architecture", "ResNet50 + Linear(2048,1)")
    mlflow.log_param("epochs", 10)
    mlflow.log_param("lambda_u", 1.0)
    mlflow.log_param("labeled_fraction", 0.05)
    mlflow.log_param("optimizer", "Adam lr=1e-3")

    mlflow.log_metric("val_auc", 0.718)
    mlflow.log_metric("val_f1", 0.66)

    mlflow.set_tag("dataset", "NIH-ChestXray14-v1.0")
    mlflow.set_tag("dvc_registry", "data_registry.md")
    mlflow.set_tag("data_split", "5pct_annote_95pct_non_annote")

    mlflow.log_artifact("model/head_semi_supervised_v1.0_auc0718.pth")

    print(f"Modèle v1.0 enregistré dans MLflow ({tracking_uri}) !")