import mlflow

mlflow.set_experiment("radio_semi_supervised")

with mlflow.start_run(run_name="v1.0"):
    mlflow.log_param("architecture", "ResNet50 + Linear(2048,1)")
    mlflow.log_param("epochs", 10)
    mlflow.log_param("lambda_u", 1.0)
    mlflow.log_param("labeled_fraction", 0.05)
    mlflow.log_param("optimizer", "Adam lr=1e-3")

    mlflow.log_metric("val_auc", 0.718)
    mlflow.log_metric("val_f1", 0.66)

    mlflow.log_artifact("model/head_semi_supervised_v1.0_auc0718.pth")

    print("✅ Modèle v1.0 enregistré dans MLflow !")
