# Contributing to Suêtes

Suêtes is research software. Contributions should preserve the connection
between scientific intent, numerical implementation, verification evidence,
and documentation.

## Development setup

```bash
git clone https://github.com/Suetes/suetes.git
cd suetes
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Before submitting a change, run:

```bash
.venv/bin/python examples/quickstart.py
.venv/bin/python -m pytest -q
.venv/bin/python -m mkdocs build --strict
.venv/bin/python -m build --wheel
```

## Change expectations

- Add or update deterministic tests for behavioral changes.
- Add a convergence or adjoint study when a regression test cannot establish
  the relevant numerical property.
- Update source docstrings when signatures, shapes, units, or return contracts
  change.
- Update narrative documentation when equations, algorithms, assumptions, or
  scientific interpretation change.
- Keep expensive model execution separate from artifact rendering.
- Do not commit generated inputs, outputs, environments, or caches.

Format changes should stay separate from scientific changes where practical.
Report the JAX backend, precision, grid, timestep, and solver settings with
numerical results.

## Reporting problems

Please include a minimal reproducer, platform, Python and JAX versions, backend
and device, precision, configuration, and the complete error message. Do not
attach restricted ERA5 or other licensed datasets to a public issue.

## Licensing of contributions

Suêtes is licensed under the Apache License 2.0. By submitting a contribution,
you agree that it may be distributed under that license and that you have the
right to submit it. Retain applicable copyright and attribution notices when
modifying existing files.
