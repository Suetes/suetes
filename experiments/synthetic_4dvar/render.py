#!/usr/bin/env python3
"""Render a synthetic 4D-Var artifact with the shared 4D-Var renderer."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parents[1] / "era5_4dvar" / "render.py", run_name="__main__")
