.PHONY: install ingest ask test evals
install:
	pip install -r requirements.txt
ingest:
	python -m graphrag ingest data/docs
ask:
	python -m graphrag ask "$(Q)"
test:
	python -m pytest -q
evals:
	python -m evals.run_evals
