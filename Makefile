.PHONY: install test lint run hud train collect-help docker

install:
	pip install -r requirements-dev.txt
	cd frontend && npm install

test:
	python -m pytest -q
	cd frontend && npm test

lint:
	ruff check .

run:  ## API + web client at http://localhost:8000/app/
	APERTURE_ENABLE_COLLECT=1 uvicorn backend.main:app --reload

hud:
	python -m backend.vision.hand_tracker

train:
	python -m backend.training.train

docker:
	docker compose up --build
