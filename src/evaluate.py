import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import mlflow
from src.utils import (
    load_config,
    load_data,
)


def create_artifact_directory(
    model_type: str,
) -> Path:

    artifact_dir = Path("artifacts") / model_type

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return artifact_dir


def main(config_path: str):

    cfg = load_config(config_path)

    model_type = cfg["model"]["type"]

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
        )
    )

    mlflow.set_experiment("evaluation")

    artifact_dir = create_artifact_directory(model_type)

    df = load_data(cfg["data"]["csv_path"])

    target = cfg["data"]["target"]

    X = df.drop(columns=[target])

    y = df[target].map(
        {
            "Yes": 1,
            "No": 0,
        }
    )

    (
        _,
        X_test,
        _,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    model = joblib.load(f"models/{model_type}.pkl")

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = model.predict(X_test)

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
    )

    plt.savefig(artifact_dir / "roc_curve.png")

    plt.close()

    # Precision Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
    )

    plt.savefig(artifact_dir / "pr_curve.png")

    plt.close()

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
    )

    plt.savefig(artifact_dir / "confusion_matrix.png")

    plt.close()

    # Classification Report
    report = classification_report(
        y_test,
        predictions,
    )

    report_path = artifact_dir / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    # Predictions
    predictions_path = artifact_dir / "predictions.csv"

    pd.DataFrame(
        {
            "actual": y_test,
            "prediction": predictions,
            "probability": probabilities,
        }
    ).to_csv(
        predictions_path,
        index=False,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    with mlflow.start_run(run_name=f"eval-{model_type}"):
        mlflow.log_param(
            "model_type",
            model_type,
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        mlflow.log_artifact(artifact_dir / "roc_curve.png")

        mlflow.log_artifact(artifact_dir / "pr_curve.png")

        mlflow.log_artifact(artifact_dir / "confusion_matrix.png")

        mlflow.log_artifact(report_path)

        mlflow.log_artifact(predictions_path)

    print("\n========== Metrics ==========")

    print(f"ROC-AUC : {roc_auc:.4f}")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    main(args.config)