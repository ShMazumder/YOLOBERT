# Makefile — one-word entry points. `make help` lists targets.
CONFIG ?= configs/base.yaml
CKPT   ?= work_dirs/base/best.pth
PY     ?= python

.PHONY: help setup train test paper clean lint tb docker

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

setup:  ## install python deps
	pip install -r requirements.txt

train:  ## train with CONFIG (override: make train CONFIG=configs/x.yaml)
	$(PY) tools/train.py --config $(CONFIG)

test:  ## evaluate CKPT on val/test
	$(PY) tools/test.py --config $(CONFIG) --checkpoint $(CKPT)

tb:  ## launch tensorboard on work_dirs
	tensorboard --logdir work_dirs/

paper:  ## compile paper/main.tex -> main.pdf
	cd paper && pdflatex -interaction=nonstopmode main.tex && \
	 bibtex main && pdflatex -interaction=nonstopmode main.tex && \
	 pdflatex -interaction=nonstopmode main.tex

lint:  ## ruff lint (pip install ruff)
	ruff check tools/ models/

docker:  ## build the image
	docker build -t yolobert:latest .

clean:  ## remove latex build junk
	cd paper && rm -f *.aux *.log *.out *.bbl *.blg *.toc
