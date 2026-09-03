# Convenience targets. Every one is a thin wrapper over the `glof` CLI so there is
# no second source of truth for how a run is invoked.
.DEFAULT_GOAL := help
PY      ?= python
CONFIG  ?= configs/toy.yaml
ENV     ?=

.PHONY: help install install-nvidia info test test-all test-nvidia lint smoke toy \
        dataset train benchmark production docker clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Toy tier: CPU-only, no NVIDIA stack
	$(PY) -m pip install -e ".[dev]"

install-nvidia:  ## Production tier: PhysicsNeMo + Earth-2 (needs torch>=2.10)
	$(PY) -m pip install -r requirements-nvidia.txt && $(PY) -m pip install -e ".[dev]"

info:  ## Resolved backends, versions and CUDA availability
	$(PY) -m glof_pipeline info

test:  ## Fast unit tests (seconds)
	$(PY) -m pytest tests -q -m "not slow" --ignore=tests/test_nvidia_integration.py

test-nvidia:  ## NVIDIA integration tests (imports and shapes only, no training)
	$(PY) -m pytest tests/test_nvidia_integration.py -q

test-all:  ## Everything, including the end-to-end smoke run
	$(PY) -m pytest tests -q

lint:  ## Style and static checks
	ruff check glof_pipeline tests && ruff format --check glof_pipeline tests

smoke:  ## Smallest end-to-end run; every stage in under a minute
	$(PY) -m glof_pipeline run --config configs/smoke.yaml

toy:  ## Full toy-tier run on CPU
	$(PY) -m glof_pipeline run --config configs/toy.yaml

dataset:  ## Build training sets for CONFIG
	$(PY) -m glof_pipeline dataset --config $(CONFIG) --which all

train:  ## Train all surrogates for CONFIG
	$(PY) -m glof_pipeline train --config $(CONFIG) --which all

benchmark:  ## Measured surrogate-vs-solver benchmark from stored checkpoints
	$(PY) -m glof_pipeline benchmark --config $(CONFIG)

production:  ## Production run (CUDA host; see docs/GCP_GPU.md)
	$(PY) -m glof_pipeline run --config configs/production.yaml --reuse-checkpoints

docker:  ## Build the CUDA container image
	docker build -t glof-twin:latest .

clean:  ## Remove caches and generated outputs
	rm -rf .pytest_cache .ruff_cache .warp_cache **/__pycache__ outputs/smoke
