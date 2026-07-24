# Rising bubble 3-D pilot

This benchmark provides the 3D rising thermal bubble test case.

Run the model:

```bash
python benchmarks/physical/rising_bubble_3d/run.py --name smoke --no-render
```

This creates `output/benchmarks/rising_bubble_3d/smoke/`, with numerical data
under `data/` and figures under `figures/`. Rerender from the saved artifact:

```bash
python benchmarks/physical/rising_bubble_3d/render.py \
  output/benchmarks/rising_bubble_3d/smoke
```

The old `--output-dir` option remains available and retains its flat-directory
behaviour for compatibility during the pilot.
