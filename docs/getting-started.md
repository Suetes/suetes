# Installation and quick start

## Requirements

Suêtes requires Python 3.10 or newer. JAX installation depends on the target
platform, particularly for NVIDIA GPU execution. Install the JAX wheel suitable
for the machine first, following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
and then install this repository.

```bash
git clone <repository-url> suetes
cd suetes
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Run the installation command from the repository root: the current directory
must contain `pyproject.toml`. Check this before installing with:

```bash
pwd
ls pyproject.toml
```

If `ls` reports that the file does not exist, change into the directory that
contains it. For example, this workspace has an outer folder and the Git
checkout nested below it, so the required command is:

```bash
cd /Users/alexbihlo/Github/suetes/suetes
python -m pip install -e .
```

The base installation contains the dependencies needed to run the model. For
development, install the `dev` extra instead; it adds pytest and the MkDocs
toolchain:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

The narrower `.[test]` and `.[docs]` extras are also available when only one
toolchain is needed.

## Check the installation

```bash
.venv/bin/python examples/quickstart.py
.venv/bin/python -m pytest -q
.venv/bin/python -m mkdocs build --strict
```

The quick start is a two-step CPU smoke simulation intended to confirm package
imports, JAX compilation, and a complete model update. Long numerical
convergence studies are intentionally outside pytest; see
[Verification and evidence](verification.md).

## Run an idealized case

```bash
.venv/bin/python benchmarks/physical/rising_bubble_3d/run.py \
  --dx 50 --dt 1 --t-end 600 \
  --output-root output --name quickstart

.venv/bin/python benchmarks/physical/rising_bubble_3d/render.py \
  output/benchmarks/rising_bubble_3d/quickstart
```

Computation and rendering are separate so figures can be regenerated from a
saved artifact without repeating an expensive integration.

## Run a configuration-driven regional case

```bash
.venv/bin/python runs/preprocess.py --config configs/wreckhouse25_config.yaml
.venv/bin/python runs/run_simulation.py --config configs/wreckhouse25_config.yaml
.venv/bin/python runs/render.py --config configs/wreckhouse25_config.yaml
```

Preprocessing requires a configured CDS API account for ERA5 downloads. Raw
and processed inputs live below `inputs/` by default; execution products live
below `output/`. See [Configuration reference](configuration.md).

## Hardware expectations

Small two-dimensional cases and reduced verification studies can run on a CPU.
Large three-dimensional real-data and differentiable experiments can require an
NVIDIA GPU with substantial device memory. JAX compilation is part of the first
execution cost, so short timing measurements should distinguish compilation
from integration.
