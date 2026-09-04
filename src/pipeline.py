from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


def build_pipeline(
    numeric_features,
    categorical_features,
    model_type="logreg",
):
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    if model_type == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            random_state=42,
        )

    else:
        raise ValueError(f"Model type non supporté : {model_type}")

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", model),
        ]
    )
