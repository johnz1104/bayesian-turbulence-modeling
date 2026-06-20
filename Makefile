# QBTM convenience Makefile.
#
# Thin wrapper over the canonical entrypoint scripts/run_tests.sh, which is the
# ONE build-and-test path shared by CI and every later agent: it configures and
# builds (cmake, including the pybind11 binding) then runs ctest + pytest.
#
# Pin the build+test interpreter with PYTHON=... (default python3); the binding
# is rebuilt against it so the ABI matches. See requirements.txt for the pins.

PYTHON ?= python3
export PYTHON

.PHONY: help test test-python test-cpp clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

test:  ## Canonical gate: configure + build binding + ctest + pytest
	scripts/run_tests.sh

test-python:  ## Python baseline only (build + pytest)
	scripts/run_tests.sh --no-cpp

test-cpp:  ## C++ baseline only (build + ctest)
	scripts/run_tests.sh --no-python

clean:  ## Remove the build tree
	rm -rf build
