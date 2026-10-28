# Common tasks. Run `make setup` once, then `make refresh` each week.
PY ?= .venv/bin/python

.PHONY: setup data live features baselines train predict optimize backtest refresh test

setup:            ## create the virtualenv and install everything
	uv venv --python 3.12 .venv || python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt -e . || uv pip install --python $(PY) -r requirements.txt -e .

data:             ## download historical seasons (vaastav) into data/raw
	$(PY) -m fplopt download-history

live:             ## fetch and cache the current season from the FPL API
	$(PY) -m fplopt fetch-live

features:         ## build data/processed/features.parquet (history + live season)
	$(PY) -m fplopt build-features

baselines:        ## evaluate baselines -> reports/phase1_baselines.md
	$(PY) -m fplopt baselines

train:            ## walk-forward evaluation + final models -> models/, reports/phase2_model.md
	$(PY) -m fplopt train

predict:          ## predict the next gameweek and list the top differentials
	$(PY) -m fplopt predict

optimize:         ## transfers for your team: make optimize TEAM=<team id> [PROJ=ep_next]
	$(PY) -m fplopt optimize $(if $(TEAM),--team-id $(TEAM)) $(if $(PROJ),--projection $(PROJ))

backtest:         ## replay past seasons per projection source -> reports/phase3_backtest.md
	$(PY) -m fplopt backtest

refresh:          ## data -> features -> baselines -> train -> predict
	$(PY) -m fplopt refresh

test:
	$(PY) -m pytest
