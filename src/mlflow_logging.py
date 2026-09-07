# src/mlflow_logging.py
import os
from pathlib import Path

import mlflow


def log_experiment():
    # Définir le nom de l'expérience
    experiment_name = "mlops-project-experiment"
    mlflow.set_experiment(experiment_name)

    # Enregistrer les métriques, paramètres et modèles
    with mlflow.start_run():
        # Exemple de métriques
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("precision", 0.92)
        mlflow.log_metric("recall", 0.89)

        # Exemple de paramètres
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("n_estimators", 100)

        # Enregistrer un modèle
        model_path = "models/random_forest.pkl"
        if os.path.exists(model_path):
            mlflow.sklearn.log_model(model_path, "random-forest-model")
        else:
            print(f"Model file not found: {model_path}")


if __name__ == "__main__":
    log_experiment()
