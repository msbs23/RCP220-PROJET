"""
Pipeline ZenML - RCP220 PROJET
Marie Soubies

Une seule commande pour tout mettre à jour :
    python pipeline.py

Ce pipeline enchaîne automatiquement :
    1. Versioning DVC  → track le modèle .pth et les données
    2. Log MLflow      → enregistre métriques + artefact dans MLflow Docker
    3. Git commit+tag  → versionne l'état complet du projet
"""

import subprocess
import os
from pathlib import Path

from zenml import pipeline, step
from zenml.client import Client

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH   = PROJECT_ROOT / "model" / "head_semi_supervised_v1.0_auc0718.pth"
DVC_FILE     = PROJECT_ROOT / "data_registry.md.dvc"
MLFLOW_URI   = "sqlite:////home/msbs/RCP_220-PROJET/mlflow_data/mlflow.db"
EXPERIMENT   = "rcp220_semi_supervised"

# Métriques du modèle actuel (à mettre à jour si vous réentraînez)
METRICS = {
    "val_auc":  0.718,
    "val_f1":   0.66,
    "lambda_u": 1.0,
    "epochs":   10,
    "lr":       1e-3,
}

MODEL_VERSION = "v1.0"

# ─── Steps ZenML ──────────────────────────────────────────────────────────────

@step
def step_dvc_versioning() -> str:
    """Versionne le modèle .pth et les données via DVC + Git."""
    print("\n[1/3] Versioning DVC...")

    # Track le modèle avec DVC s'il n'est pas déjà tracké
    model_dvc = PROJECT_ROOT / "model" / "head_semi_supervised_v1.0_auc0718.pth.dvc"
    if not model_dvc.exists():
        subprocess.run(
            ["dvc", "add", str(MODEL_PATH)],
            cwd=PROJECT_ROOT, check=True
        )
        print(f"  ✓ Modèle ajouté à DVC : {MODEL_PATH.name}")
    else:
        print(f"  ✓ Modèle déjà tracké par DVC")

    # Récupère le hash DVC du fichier de données
    result = subprocess.run(
        ["git", "log", "--oneline", "-1", "data_registry.md.dvc"],
        cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    dvc_commit = result.stdout.strip() or "non versionné"
    print(f"  ✓ Hash DVC données : {dvc_commit}")

    return dvc_commit


@step
def step_mlflow_log(dvc_commit: str) -> str:
    """Log les métriques et l'artefact modèle dans MLflow Docker."""
    import mlflow

    print("\n[2/3] Logging MLflow...")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"pipeline_{MODEL_VERSION}") as run:
        # Hyperparamètres
        mlflow.log_param("lambda_u",       METRICS["lambda_u"])
        mlflow.log_param("epochs",         METRICS["epochs"])
        mlflow.log_param("lr",             METRICS["lr"])
        mlflow.log_param("backbone",       "ResNet50-frozen")
        mlflow.log_param("labeled_ratio",  "5%")

        # Métriques
        mlflow.log_metric("val_auc", METRICS["val_auc"])
        mlflow.log_metric("val_f1",  METRICS["val_f1"])

        # Tags traçabilité
        mlflow.set_tag("model_version", MODEL_VERSION)
        mlflow.set_tag("dvc_commit",    dvc_commit)
        mlflow.set_tag("pipeline",      "zenml")
        mlflow.set_tag("source",        "pipeline.py")

        # Artefact modèle
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="../model")

        run_id = run.info.run_id
        print(f"  ✓ Run MLflow : {run_id}")
        print(f"  ✓ AUC={METRICS['val_auc']} | F1={METRICS['val_f1']}")
        print(f"  ✓ Artefact loggé : {MODEL_PATH.name}")

    return run_id


@step
def step_git_commit(run_id: str, dvc_commit: str) -> None:
    """Commit et tag Git pour versionner l'état complet du projet."""
    print("\n[3/3] Git commit + tag...")

    # Ajoute les fichiers .dvc modifiés
    subprocess.run(
        ["git", "add", "*.dvc", "model/*.dvc", ".dvc/", "data_registry.md.dvc"],
        cwd=PROJECT_ROOT, check=False  # check=False car certains peuvent ne pas exister
    )

    # Commit
    commit_msg = (
        f"pipeline: {MODEL_VERSION} | "
        f"AUC={METRICS['val_auc']} F1={METRICS['val_f1']} | "
        f"mlflow_run={run_id[:8]}"
    )
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✓ Commit : {commit_msg}")
    else:
        print(f"  ℹ Rien à committer (état déjà versionné)")

    # Tag Git avec la version du modèle
    tag_name = f"model-{MODEL_VERSION}-auc{int(METRICS['val_auc']*1000)}"
    subprocess.run(
        ["git", "tag", "-f", tag_name],
        cwd=PROJECT_ROOT, check=True
    )
    print(f"  ✓ Tag Git : {tag_name}")
    print(f"\n✅ Pipeline terminée avec succès !")
    print(f"   → MLflow run : {run_id}")
    print(f"   → Git tag    : {tag_name}")
    print(f"   → Voir MLflow: http://localhost:5000")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

@pipeline(name="rcp220_update_pipeline")
def rcp220_pipeline():
    """
    Pipeline de mise à jour RCP220.
    Enchaîne DVC → MLflow → Git en une seule commande.
    """
    dvc_commit = step_dvc_versioning()
    run_id     = step_mlflow_log(dvc_commit)
    step_git_commit(run_id, dvc_commit)


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Vérifie que le stack ZenML est bien configuré
    client = Client()
    active_stack = client.active_stack_model
    print(f"Stack actif : {active_stack.name}")
    print(f"Démarrage de la pipeline RCP220...\n")

    rcp220_pipeline()