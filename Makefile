.PHONY: install train test run-api run-dashboard simulate docker-up docker-down clean

PYTHON = ./.venv/Scripts/python
PIP = ./.venv/Scripts/pip

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) -m src.models.train

test:
	$(PYTHON) -m pytest -v

run-api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

simulate:
	$(PYTHON) -m streaming.producer --rate 2.0 --count 30

docker-up:
	docker compose up --build -d

docker-up-streaming:
	docker compose --profile streaming up --build -d

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ .pytest_cache .coverage mlruns
