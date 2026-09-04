import joblib
import matplotlib.pyplot as plt

model = joblib.load("models/random_forest.pkl")

rf = model.named_steps["model"]

importances = rf.feature_importances_

plt.figure(figsize=(10, 5))

plt.bar(
    range(len(importances)),
    importances,
)

plt.title("Random Forest Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

print("feature_importance.png generated")
