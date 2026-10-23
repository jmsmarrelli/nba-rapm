# Pipeline stages share generated inputs and must run in order.
.NOTPARALLEL:
PY ?= python3

.PHONY: all stints rapm validate boxscore clean site

all: verify-data stints rapm validate boxscore

stints:
	$(PY) src/run_01_stints.py

rapm:
	$(PY) src/run_02_rapm.py

validate:
	$(PY) src/run_03_validate.py

boxscore:
	$(PY) src/run_04_boxscore.py

# Regenerate the GitHub Pages page from README.md. Needs `pip install -r requirements.txt`.
site:
	$(PY) tools/site.py

clean:
	rm -f figures/*.png output/*.csv data/processed/*

.PHONY: test
test:
	$(PY) -m pytest -q tests

.PHONY: fetch-data verify-data
fetch-data:
	$(PY) tools/fetch_data.py

verify-data:
	$(PY) tools/fetch_data.py --check

.PHONY: report
report:
	$(PY) tools/report.py
