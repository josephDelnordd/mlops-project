import argparse
import os

import joblib
import mlflow
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


def main(config_path):

    cfg = load_config(config_path)

    df = load_data(cfg["data"]["csv_path"])

    target = cfg["data"]["target"]

    X = df.drop(columns=[target])

    y = df[target].map(
        {
            "Yes": 1,
            "No": 0,
        }
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    print("\nTypes numériques :")
    print(X_train[cfg["features"]["numeric"]].dtypes)

    pipeline = build_pipeline(
        cfg["features"]["numeric"],
        cfg["features"]["categorical"],
        cfg["model"]["type"],
    )

    cv = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=True,
        random_state=42,
    )

    mlflow.set_experiment("churn-exp")

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["model"]["params"],
            cv=cv,
            scoring=cfg["cv"]["scoring"],
            n_jobs=-1,
            error_score="raise",
        )

        grid.fit(
            X_train,
            y_train,
        )

        best_model = grid.best_estimator_

        probs = best_model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(
            y_test,
            probs,
        )

        mlflow.log_metric(
            "test_auc",
            auc,
        )

        os.makedirs("models", exist_ok=True)

        model_type = cfg["model"]["type"]

        joblib.dump(best_model, f"models/{model_type}.pkl")

        print(f"\nBest Params : {grid.best_params_}")

        print(f"ROC-AUC Test : {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    main(args.config)
