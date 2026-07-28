# NEUVE reachability diagnostic

`experiments/neuve_coordinates/diagnostics/reachability_pipeline.py` diagnoses whether a
poor PGF-trained NEUVE result is caused by insufficient representational
capacity or by optimization in neural-network weight space.

The command requires an existing training dataset and the SLEVE configuration
tuned on that same dataset:

```bash
MPLCONFIGDIR=/tmp/suetes-mpl \
.venv/bin/python3 experiments/neuve_coordinates/diagnostics/reachability_pipeline.py \
  --dataset output/neuve_pgf/training_dataset.json \
  --sleve-config output/neuve_pgf/best_sleve.json \
  --output-dir output/neuve_pgf_reachability
```

The four stages are:

1. fit the original terrain-conditioned MLP to the tuned SLEVE grid as a
   capacity audit;
2. optimize a scalar full-horizon aggressiveness parameter within the
   geometrically admissible range;
3. expand that solution into 16 vertical density coefficients and refine the
   complete profile using the full integration;
4. introduce and optimize a bounded terrain-conditioned residual.

The first stage is diagnostic only. Its SLEVE-fitted weights never initialize
the self-supervised stages.

For a quick pipeline check before a full run:

```bash
MPLCONFIGDIR=/tmp/suetes-mpl \
.venv/bin/python3 experiments/neuve_coordinates/diagnostics/reachability_pipeline.py \
  --dataset output/neuve_pgf/training_dataset.json \
  --sleve-config output/neuve_pgf/best_sleve.json \
  --output-dir output/neuve_pgf_reachability_pilot \
  --audit-epochs 30 \
  --scalar-epochs 10 \
  --shape-epochs 10 \
  --conditioned-epochs 10
```

The output directory contains separate checkpoints for all four stages,
training histories, a JSON comparison against Gal--Chen and tuned SLEVE, and
`reachability_pipeline.png`.

## Full-horizon aggressiveness scan

When gradient training remains in the Gal--Chen-like basin, scan the family

\[
\log \rho(\eta)=a(1-2\eta)
\]

with isolated forward-model workers:

```bash
MPLCONFIGDIR=/tmp/suetes-mpl \
.venv/bin/python3 experiments/neuve_coordinates/diagnostics/aggressiveness_scan.py \
  --dataset output/neuve_pgf/training_dataset.json \
  --sleve-config output/neuve_pgf/best_sleve.json \
  --output-dir output/neuve_pgf_aggressiveness_scan
```

At \(a=0\), the direct-density coordinate is exactly Gal--Chen. Increasing
\(a\) moves progressively toward rapid terrain decay. The resulting CSV, JSON,
and figure show the full-horizon TKE and minimum layer thickness alongside
Gal--Chen and tuned SLEVE.

## Scalar-gradient audit

If gradient optimization disagrees with the forward loss scan, compare the
reverse derivative with centered finite differences:

```bash
MPLCONFIGDIR=/tmp/suetes-mpl \
.venv/bin/python3 experiments/neuve_coordinates/diagnostics/scalar_gradient_audit.py \
  --dataset output/neuve_pgf/training_dataset.json \
  --output-dir output/neuve_pgf_scalar_gradient_audit
```

The default audit checks \(a=0.5\) over 25, 50, 100, 250, and 500 steps, plus
\(a=0,0.5,1,2\) at the full horizon. Every case runs in an isolated process
and is checkpointed independently.

Each stage clears the JAX executable cache before the next reverse graph is
compiled. If a CUDA or XLA process failure still interrupts a run, resume in a
fresh process from the next saved stage. For example:

```bash
MPLCONFIGDIR=/tmp/suetes-mpl \
.venv/bin/python3 experiments/neuve_coordinates/diagnostics/reachability_pipeline.py \
  --dataset output/neuve_pgf/training_dataset.json \
  --sleve-config output/neuve_pgf/best_sleve.json \
  --output-dir output/neuve_pgf_reachability_pilot \
  --start-stage 4 \
  --conditioned-epochs 10
```
