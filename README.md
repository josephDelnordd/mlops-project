# 🚀 Churn Classifier with MLOps

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

## 📦 Architecture

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
│   ├── mlflow_logging.py
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

## 🧠 Modèles

### Logistic Regression

- ROC-AUC : **0.8382**

### Random Forest

- ROC-AUC : **0.8418**
- Accuracy : **0.7928**
- F1 Score : **0.5350**

**Meilleur modèle : Random Forest**

---

## 🛠 Fonctionnalités

- Prétraitement automatique des données
- Encodage des variables catégorielles
- Normalisation des variables numériques
- Validation croisée (StratifiedKFold)
- Optimisation des hyperparamètres (GridSearchCV)
- Suivi des expériences avec **MLflow**
- Génération automatique des artefacts d'évaluation

---

🧰 Prérequis

- Docker et Docker Compose installés
- Python 3.12
- Outils nécessaires : pip, pytest, ruff

---

## ✅ Installation

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

# Build le projet avec Docker
make build

# Lancer le conteneur Docker
make up
```

---

🧪 Exécution des tests

```bash
make test
```

---

## 🛠 Linting et formatage du code

```bash
make format
make lint
```

---

## 🧬 Entraînement des modèles

### Logistic Regression

```bash
make train-logreg
```

### Random Forest

```bash
make train-random-forest
```

Les modèles entraînés sont sauvegardés dans le répertoire `models/`.


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

Les résultats d'évaluation (métriques, courbes ROC, PR, confusion matrices, etc.) sont générés dans le répertoire `artifacts/`.

---

## ⚡ Entraînement & évaluation rapides

```bash
make build
make up
make all
```

---

## 📦 MLflow Tracking

MLflow est utilisé pour suivre les expériences et enregistrer les artefacts. Les logs et les métriques sont disponibles via l'interface web de MLflow.

---

## 🖥 Accès à MLflow

Ouvrez votre navigateur et rendez-vous à l'adresse :

```bash
http://localhost:5000
```

---

## 🧹 Nettoyage

Pour arrêter et supprimer tous les conteneurs et volumes :

```bash
make down
```

---

## 🧪 Qualité et Sécurité

Le projet intègre des outils de qualité et de sécurité :

- **Pytest :** Tests unitaires et fonctionnels.
- **Ruff :** Linting et formatage du code.
- **GitHub Actions :** CI/CD automatisé.
- **pip-audit :** Vérification des vulnérabilités des dépendances.
- **Trivy :** Analyse des images Docker pour les vulnérabilités.

---

## 📄 Licence
Ce projet est sous licence MIT.

---

## 🧑‍💻 Auteur

Ce projet a été développé par Joseph DELNORD.