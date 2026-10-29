PY := .venv/bin/python

.PHONY: install test test-fast acceptance data baseline text text-quick url fusion evaluate serve

install:
	python3.12 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

acceptance:
	PHISHGUARD_ACCEPTANCE=1 $(PY) -m pytest -m acceptance

data:
	$(PY) -m phishguard prepare-data

baseline:
	$(PY) -m phishguard train-baseline

text-quick:
	$(PY) -m phishguard train-text --max-train-samples 20000

text:
	$(PY) -m phishguard train-text

url:
	$(PY) -m phishguard train-url

fusion:
	$(PY) -m phishguard fit-fusion

evaluate:
	$(PY) -m phishguard evaluate

serve:
	$(PY) -m phishguard serve
