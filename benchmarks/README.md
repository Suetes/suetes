# Benchmarks

Reusable reference problems and performance measurements live here.

- `physical/` contains recognized model test cases such as the rising thermal
  bubble and Schär mountain-wave problem.
- `performance/` contains runtime, memory, scaling, and throughput benchmarks.

Benchmarks may produce plots or timing data and do not need to expose a
pass/fail result. Numerical convergence and equation-consistency programs
belong in `verification/`; paper-specific science belongs in `experiments/`.

## Artifact workflow

Benchmark entry points use the common execution-bundle layout:

```text
output/benchmarks/<case>/<name>/
├── data/
│   ├── artifact.nc, or structured CSV data
│   └── artifact.json
└── figures/
```

Use `--name` to keep parameter studies separate, `--output-root` to place all
bundles on another filesystem, or the compatibility option `--output-dir` for
an explicit flat directory. Simulation entry points with figures support
`--no-render`; the corresponding renderer consumes the saved bundle without
rerunning the model.

For example:

```bash
python benchmarks/physical/rising_bubble_3d.py --name paper --no-render
python benchmarks/physical/rising_bubble_3d/render.py \
  output/benchmarks/rising_bubble_3d/paper
```

Case-specific renderers own scientific figure composition. Reusable maps,
sections, spectra, and styling remain in `suetes.vis`.
