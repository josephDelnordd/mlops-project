import pandas as pd

from src.pipeline import (
    build_pipeline,
)


def test_build_pipeline():

    pipe = build_pipeline(
        ["tenure"],
        ["gender"],
        "logreg",
    )

    assert "pre" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_fit_pipeline():

    X = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4],
            "gender": [
                "Male",
                "Female",
                "Male",
                "Female",
            ],
        }
    )

    y = [0, 1, 0, 1]

    pipe = build_pipeline(
        ["tenure"],
        ["gender"],
        "logreg",
    )

    pipe.fit(
        X,
        y,
    )

    preds = pipe.predict(X)

    assert len(preds) == 4
