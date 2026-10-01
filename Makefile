.PHONY: install ingest api test eval lint clean

install:
	pip install -r requirements.txt

ingest:
	python scripts/ingest.py

api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/

eval:
	python scripts/run_evaluation.py

seed-langsmith:
	python scripts/seed_langsmith.py

clean:
	rm -rf .pytest_cache __pycache__ data/chroma data/checkpoints.sqlite .smoke_data
	find . -name "*.pyc" -delete