# Makefile - common developer tasks

.PHONY: help lint test build sdist wheel clean

export SPHINX_MOCK_REQUIREMENTS=1
export IT_USE_CT_COMMIT_PIN="1"

help:
	@echo "Makefile commands:"
	@echo "  make clean      - remove build artifacts"
	@echo "  make lint       - run ruff/pre-commit lint checks (local)"
	@echo "  make test       - run pytest (requires dev deps)"
	@echo "  make docs       - build documentation"

clean:
	rm -rf build dist *.egg-info .pytest_cache
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf $(shell find . -name "lightning_log")
	rm -rf $(shell find . -name "lightning_logs")
	rm -rf .ruff_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api

lint:
	# run ruff via pre-commit where available
	pre-commit run ruff-check --all-files
	pre-commit run ruff-format --all-files

test: clean
	pip install -r requirements/devel.txt
	# TODO: remove this once no longer necessary
	pip install --upgrade -r requirements/post_upgrades.txt
	# TODO: remove this once no longer necessary
	@echo "Using IT_USE_CT_COMMIT_PIN for circuit-tracer installation"
	interpretune-install-circuit-tracer
	# run tests with coverage (cpu-only, running gpu standalone tests required for full coverage)
	python -m coverage run --append --source src/interpretune -m pytest src/interpretune tests -v
	python -m coverage report

docs: clean
	pip install --quiet -r requirements/docs.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build
