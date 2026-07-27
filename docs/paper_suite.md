# Reproducing the manuscript experiments

`paper_suite.toml` is the authoritative list of simulations, inversions,
sensitivities, and verification studies used by the paper. The runner stores
all new results below `output/paper/`, including numerical artifacts, figures,
logs, the exact manifest, its SHA-256 digest, and the Git commit.

List or inspect the suite without running anything:

```bash
.venv/bin/python3 scripts/run_paper_suite.py list
.venv/bin/python3 scripts/run_paper_suite.py status
.venv/bin/python3 scripts/run_paper_suite.py all --dry-run
```

Run one case, all table cases, or all appendix cases:

```bash
.venv/bin/python3 scripts/run_paper_suite.py all rising_bubble
.venv/bin/python3 scripts/run_paper_suite.py all --group table
.venv/bin/python3 scripts/run_paper_suite.py all --group appendix
```

The `run` and `render` actions can be separated:

```bash
.venv/bin/python3 scripts/run_paper_suite.py run squall_forward
.venv/bin/python3 scripts/run_paper_suite.py render squall_forward
```

Stages whose declared outputs already exist are skipped. Pass `--force` to
rerun them. A failed command stops the suite immediately and leaves its full
console output in `output/paper/logs/<case>/<stage>.log`.

To add or remove a paper experiment, edit only `paper_suite.toml`. Commands
are stored as TOML arrays and are executed without a shell. This avoids
quoting ambiguity and ensures that the recorded command is the command that
was actually run.
