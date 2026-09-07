COMPOSE=docker compose

.PHONY: help \
	build rebuild up down \
	mlflow mlflow-logs mlflow-stop \
	train-logreg train-random-forest \
	evaluate-logreg evaluate-random-forest \
	test lint format \
	clean clean-all all

help:
	@echo ""
	@echo "========== MLOps Project =========="
	@echo ""
	@echo "Infrastructure"
	@echo "  make build                  Build Docker images"
	@echo "  make rebuild                Rebuild all images"
	@echo "  make up                     Start services"
	@echo "  make down                   Stop services"
	@echo ""
	@echo "MLflow"
	@echo "  make mlflow                 Start MLflow server"
	@echo "  make mlflow-logs            Show MLflow logs"
	@echo "  make mlflow-stop            Stop MLflow server"
	@echo ""
	@echo "Training"
	@echo "  make train-logreg           Train Logistic Regression"
	@echo "  make train-random-forest    Train Random Forest"
	@echo ""
	@echo "Evaluation"
	@echo "  make evaluate-logreg        Evaluate Logistic Regression"
	@echo "  make evaluate-random-forest Evaluate Random Forest"
	@echo ""
	@echo "Quality"
	@echo "  make test                   Run Pytest"
	@echo "  make lint                   Run Ruff"
	@echo "  make format                 Format code"
	@echo ""
	@echo "Cleaning"
	@echo "  make clean                  Remove artifacts & models"
	@echo "  make clean-all              Remove MLflow data too"
	@echo ""
	@echo "Workflow"
	@echo "  make all                    Train and evaluate all models"
	@echo ""

build:
	$(COMPOSE) build --no-cache

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down --remove-orphans

rebuild:
	$(MAKE) down
	$(MAKE) build

#
# MLFLOW
#

mlflow:
	$(COMPOSE) up -d mlflow

run-mlflow:
	python -m src.mlflow_logging

mlflow-logs:
	$(COMPOSE) logs -f mlflow

mlflow-stop:
	$(COMPOSE) stop mlflow

#
# TRAINING
#

train-logreg:
	$(COMPOSE) run --rm train-logreg

train-random-forest:
	$(COMPOSE) run --rm train-random-forest

#
# EVALUATION
#

evaluate-logreg:
	$(COMPOSE) run --rm evaluate-logreg

evaluate-random-forest:
	$(COMPOSE) run --rm evaluate-random-forest

#
# TESTS / QUALITY
#

test:
	$(COMPOSE) run --rm tests

lint:
	$(COMPOSE) run --rm lint

format:
	$(COMPOSE) run --rm format

#
# CLEANUP
#

clean:
	rm -rf artifacts/*
	rm -rf models/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

clean-all: clean
	rm -rf mlflow
	rm -rf mlruns
	rm -f mlflow.db

#
# FULL WORKFLOW
#

all:
	$(MAKE) mlflow

	$(MAKE) train-logreg
	$(MAKE) evaluate-logreg

	$(MAKE) train-random-forest
	$(MAKE) evaluate-random-forest