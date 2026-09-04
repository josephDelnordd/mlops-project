import argparse
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
from sklearn.model_selection import (
    train_test_split,
)

from src.utils import (
    load_config,
    load_data,
)


def create_artifact_directory(
    model_type: str,
) -> Path:
    """
    Create artifact directory.

    Example:
    artifacts/logreg/
    artifacts/random_forest/
    """

    artifact_dir = Path("artifacts") / model_type

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return artifact_dir


def save_roc_curve(
    y_true,
    probabilities,
    output_path,
):
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()


def save_pr_curve(
    y_true,
    probabilities,
    output_path,
):
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()


def save_confusion_matrix(
    y_true,
    predictions,
    output_path,
):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()


def save_classification_report(
    y_true,
    predictions,
    output_path,
):
    report = classification_report(
        y_true,
        predictions,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    return report


def save_predictions(
    y_true,
    predictions,
    probabilities,
    output_path,
):
    results = pd.DataFrame(
        {
            "actual": y_true,
            "prediction": predictions,
            "probability": probabilities,
        }
    )

    results.to_csv(
        output_path,
        index=False,
    )


def evaluate_model(
    model,
    X_test,
):
    probabilities = (model.predict_proba(X_test))[:, 1]

    predictions = model.predict(X_test)

    return (
        predictions,
        probabilities,
    )


def print_metrics(
    y_test,
    predictions,
    probabilities,
):
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

    print("\n========== Metrics ==========")

    print(f"ROC-AUC : {roc_auc:.4f}")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"F1 Score : {f1:.4f}")


def main(
    config_path: str,
):

    cfg = load_config(config_path)

    model_type = cfg["model"]["type"]

    artifact_dir = create_artifact_directory(model_type)

    print(f"\nArtifacts directory: {artifact_dir}")

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

    model_path = Path("models") / f"{model_type}.pkl"

    model = joblib.load(model_path)

    (
        predictions,
        probabilities,
    ) = evaluate_model(
        model,
        X_test,
    )

    save_roc_curve(
        y_test,
        probabilities,
        artifact_dir / "roc_curve.png",
    )

    save_pr_curve(
        y_test,
        probabilities,
        artifact_dir / "pr_curve.png",
    )

    save_confusion_matrix(
        y_test,
        predictions,
        artifact_dir / "confusion_matrix.png",
    )

    report = save_classification_report(
        y_test,
        predictions,
        artifact_dir / "classification_report.txt",
    )

    save_predictions(
        y_test,
        predictions,
        probabilities,
        artifact_dir / "predictions.csv",
    )

    print(report)

    print_metrics(
        y_test,
        predictions,
        probabilities,
    )

    print("\nGenerated files:")

    for file in sorted(artifact_dir.iterdir()):
        print(f"  - {file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Evaluate churn model"))

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML config",
    )

    args = parser.parse_args()

    main(args.config)
