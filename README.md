# Churn Classifier with MLOps

Projet de classification du churn client basé sur le dataset **Telco Customer Churn**.

L'objectif est de mettre en œuvre un pipeline Machine Learning reproductible avec des bonnes pratiques **MLOps** :

- Scikit-Learn Pipeline
- ColumnTransformer
- GridSearchCV
- MLflow
- Docker / Docker Compose
- Pytest
- Ruff
- GitHub Actions
- DevSecOps (pip-audit, Trivy)

---

## Architecture

```text
mlops-project/
├── artifacts/
├── configs
│   ├── logreg.yaml
│   └── random_forest.yaml
├── data
│   └── raw.csv
├── models/
├── src
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── pipeline.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Modèles

### Logistic Regression

- ROC-AUC : **0.8382**

### Random Forest

- ROC-AUC : **0.8418**
- Accuracy : **0.7928**
- F1 Score : **0.5350**

**Meilleur modèle : Random Forest**

---

## Fonctionnalités

- Prétraitement automatique des données
- Encodage des variables catégorielles
- Normalisation des variables numériques
- Validation croisée (StratifiedKFold)
- Optimisation des hyperparamètres (GridSearchCV)
- Suivi des expériences avec MLflow
- Génération automatique des artefacts d'évaluation

---

## Installation

```bash
# Cloner le dépôt et se placer dans le répertoire du projet
git clone <repository-url>
cd mlops-project

# Copier le fichier d'environnement
cp .env.example .env

# Installation et activation de l'environnement locale
python -m venv .venv
source .venv/bin/activate

# Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Entraînement

### Logistic Regression

```bash
make train-logreg
```

### Random Forest

```bash
make train-random-forest
```

---

## Évaluation

### Logistic Regression

```bash
make evaluate-logreg
```

### Random Forest

```bash
make evaluate-random-forest
```

Les résultats sont générés dans :

```text
artifacts/
├── logreg/
└── random_forest/
```

---

## Qualité

Tests :

```bash
make test
```

Lint :

```bash
make lint
```

Formatage :

```bash
make format
```

---

## Docker

Build :

```bash
make build
```

Train :

```bash
make train-logreg
make train-random-forest

```

Évaluation :

```bash
make evaluate-random-forest
make evaluate-logreg
```

---

## CI/CD & Sécurité

Le pipeline GitHub Actions exécute automatiquement :

- Ruff
- Pytest
- Build Docker
- pip-audit

