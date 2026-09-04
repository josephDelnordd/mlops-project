COMPOSE=docker compose

.PHONY: help build clean \
	train-logreg train-random-forest \
	evaluate-logreg evaluate-random-forest \
	test lint down

help:
	@echo ""
	@echo "=== Available commands ==="
	@echo ""
	@echo "build                 Build Docker images"
	@echo "train-logreg          Train Logistic Regression model"
	@echo "train-random-forest   Train Random Forest model"
	@echo "evaluate-logreg       Evaluate Logistic Regression"
	@echo "evaluate-random-forest Evaluate Random Forest"
	@echo "test                  Run pytest"
	@echo "lint                  Run ruff"
	@echo "down                  Stop containers"
	@echo "clean                 Remove generated artifacts"
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

all:
	$(MAKE) train-logreg
	$(MAKE) evaluate-logreg
	$(MAKE) train-random-forest
	$(MAKE) evaluate-random-forest

train-logreg:
	$(COMPOSE) run --rm train-logreg

train-random-forest:
	$(COMPOSE) run --rm train-random-forest

evaluate-logreg:
	$(COMPOSE) run --rm evaluate-logreg

evaluate-random-forest:
	$(COMPOSE) run --rm evaluate-random-forest

test:
	$(COMPOSE) run --rm tests

lint:
	$(COMPOSE) run --rm tests ruff check .

format:
	$(COMPOSE) run --rm tests sh -c "ruff check . --fix && ruff format ."

clean:
	rm -rf artifacts/*
	rm -rf models/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

clean-all: clean
	rm -rf mlruns
	rm -f mlflow.db