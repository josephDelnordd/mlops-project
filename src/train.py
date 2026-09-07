import argparse
import os

import joblib
import mlflow.sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src.pipeline import build_pipeline
from src.utils import (
    load_config,
    load_data,
)


def main(config_path: str):
    # Charger la configuration
    cfg = load_config(config_path)

    # Configurer MLflow
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000",
        )
    )
    mlflow.set_experiment(
        os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            "churn-exp",
        )
    )

    # Charger les données
    df = load_data(cfg["data"]["csv_path"])

    target = cfg["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target].map({"Yes": 1, "No": 0})

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    # Construire le pipeline
    pipeline = build_pipeline(
        cfg["features"]["numeric"],
        cfg["features"]["categorical"],
        cfg["model"]["type"],
    )

    # Configurer la validation croisée
    cv = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=True,
        random_state=42,
    )

    # Activer l'auto-logging pour MLflow
    mlflow.sklearn.autolog()

    model_type = cfg["model"]["type"]

    with mlflow.start_run(run_name=f"train-{model_type}"):
        # Effectuer la recherche de grille
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["model"]["params"],
            cv=cv,
            scoring=cfg["cv"]["scoring"],
            n_jobs=-1,
            error_score="raise",
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Évaluer le modèle sur les données de test
        probabilities = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)

        # Enregistrer les métriques dans MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("test_auc", auc)

        # Sauvegarder le modèle localement
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_type}.pkl"
        joblib.dump(best_model, model_path)

        # Enregistrer le modèle comme artefact dans MLflow
        mlflow.log_artifact(model_path, artifact_path="model")

        print(f"\nModel saved : {model_path}")
        print(f"Best Params : {grid.best_params_}")
        print(f"ROC-AUC Test : {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    main(args.config)
