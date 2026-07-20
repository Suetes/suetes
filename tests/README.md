# Automated tests

This directory contains deterministic pytest regression tests with explicit
assertions. It is the only directory collected automatically by pytest, as
configured in `pyproject.toml`.

Long-running convergence studies belong in `verification/`. Physical and
performance reference cases belong in `benchmarks/`.

