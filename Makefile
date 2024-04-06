.PHONY: help install

help:
	@cat $(firstword $(MAKEFILE_LIST))

install:
	python -m poetry install --no-root
