.PHONY: help info install run

PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

help:
	@cat $(firstword $(MAKEFILE_LIST))

info:
	nvidia-smi
	nvcc --version
	poetry which python
	poetry run python --version

install:
	python -m poetry install --no-root

run:
	time poetry run python main.py
