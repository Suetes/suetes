#!/usr/bin/env python3
"""Create a reproducible terrain dataset for coordinate experiments."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments._shared.neuve_coordinate import TARGET_DEFAULTS, make_dataset, parse_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain-family", choices=["ridge", "random3d", "multiscale3d"], default="multiscale3d")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--role", choices=["training", "testing"], required=True)
    parser.add_argument("--target", choices=["pgf_rest", "tracer_reversibility", "mountain_flux"], default="pgf_rest")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    steps = args.steps or TARGET_DEFAULTS[args.target]["steps"]
    dataset = make_dataset(args.terrain_family, parse_seeds(args.seeds), args.role, args.output, target=args.target, steps=steps)
    print(f"Saved {args.role} dataset with {len(dataset['seeds'])} terrains to {args.output}")


if __name__ == "__main__":
    main()
