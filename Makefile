.PHONY: test lint type-check format docker-up docker-down dashboard api setup clean

test:
	pytest tests/ -x --tb=short -q

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --tb=short

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

dashboard:
	streamlit run dashboard/app.py

api:
	uvicorn src.api.main:app --reload --port 8000

xai:
	python scripts/run_xai_pipeline.py

setup:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
