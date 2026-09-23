PY := python3

.PHONY: all stints rapm validate boxscore clean site

all: stints rapm validate boxscore

stints:
	$(PY) src/run_01_stints.py

rapm:
	$(PY) src/run_02_rapm.py

validate:
	$(PY) src/run_03_validate.py

boxscore:
	$(PY) src/run_04_boxscore.py

# Regenerate the GitHub Pages page from README.md. Needs `pip install markdown`.
site:
	$(PY) tools/site.py

clean:
	rm -f figures/*.png output/*.csv data/processed/*
