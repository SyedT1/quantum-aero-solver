"""Command-line entry point for corrected, reproducible challenge experiments."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np

from quantum_aero.carleman import exact_bgk, order2_collision, validate_block_encoding
from quantum_aero.classical import LBMConfig, W, relative_l2, run_lbm
from quantum_aero.io import write_dataset
from quantum_aero.quantum import (
    applied_noise_experiment,
    transpiled_collision_resources,
    transpiled_streaming_resources,
)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def validate() -> None:
    output = Path("results")
    cfg = LBMConfig(n=32, reynolds=100, t_end=0.1, snapshots=6)
    exact = run_lbm(cfg, keep_fields=True)
    carleman = run_lbm(cfg, collision=order2_collision, keep_fields=True)
    exact_fields = exact.pop("fields")
    carleman_fields = carleman.pop("fields")
    rng = np.random.default_rng(7)
    velocity = rng.uniform(-0.03, 0.03, 2)
    cu = velocity[0] * np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cu += velocity[1] * np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    f = W * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * np.dot(velocity, velocity))
    block = validate_block_encoding(f, omega=1.2)
    collision_error = np.linalg.norm(order2_collision(f, 1.2) - exact_bgk(f, 1.2))
    payload = {
        "full_spatial_exact_bgk": exact,
        "full_spatial_order2_carleman": carleman,
        "final_velocity_relative_l2_carleman_vs_bgk": relative_l2(
            carleman_fields[-1]["u"], carleman_fields[-1]["v"],
            exact_fields[-1]["u"], exact_fields[-1]["v"],
        ),
        "final_population_relative_l2_carleman_vs_bgk": float(
            np.linalg.norm(carleman_fields[-1]["f"] - exact_fields[-1]["f"])
            / np.linalg.norm(exact_fields[-1]["f"])
        ),
        "single_collision_absolute_error": float(collision_error),
        "block_encoding": block,
        "collision_resources": transpiled_collision_resources(),
        "streaming_resources": transpiled_streaming_resources(),
        "noise": applied_noise_experiment(),
        "scope_warning": (
            "The spatial Carleman run rebuilds f tensor f classically each step. "
            "It validates collision physics but is not an end-to-end FTQC speedup claim."
        ),
    }
    dump_json(output / "validation.json", payload)
    print(json.dumps(payload, indent=2))


def sweep(args: argparse.Namespace) -> None:
    pairs = [(10, 32), (100, 64), (400, 128), (1000, 256), (5000, args.n5000)]
    rows = []
    for reynolds, n in pairs:
        trials = []
        result = None
        for _ in range(args.repeats):
            result = run_lbm(
                LBMConfig(n=n, reynolds=reynolds, t_end=args.t_end, snapshots=2)
            )
            trials.append(result["runtime_seconds"])
        assert result is not None
        final = result["records"][-1]
        rows.append(
            {
                "reynolds": reynolds,
                "n": n,
                "t_end": args.t_end,
                "steps": result["steps"],
                "runtime_median_seconds": float(np.median(trials)),
                "runtime_min_seconds": float(np.min(trials)),
                "runtime_max_seconds": float(np.max(trials)),
                "relative_l2": final["relative_l2"],
                "vortex_relative_l2": final["vortex_relative_l2"],
                "kinetic_energy_error": abs(
                    final["kinetic_energy"] - final["exact_kinetic_energy"]
                ),
                "divergence_l2": final["divergence_l2"],
                "memory_bytes": result["population_memory_bytes"],
                "resolution_status": (
                    "measured pilot; grid convergence not established"
                    if reynolds == 5000 else "configured scaling point"
                ),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "processor": platform.processor(),
            }
        )
        print(json.dumps(rows[-1]))
    output = Path("results/reynolds_sweep.csv")
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def datasets(args: argparse.Namespace) -> None:
    for reynolds, n in [(10, 32), (100, 64), (400, 128), (1000, 256)]:
        target = Path("baseline/dataset") / f"lbm_Re{reynolds}_N{n}.h5"
        result = write_dataset(
            target, LBMConfig(n=n, reynolds=reynolds, t_end=args.t_end, snapshots=21)
        )
        print(target, result["records"][-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    sweep_parser = commands.add_parser("sweep")
    sweep_parser.add_argument("--t-end", type=float, default=1.0)
    sweep_parser.add_argument("--n5000", type=int, default=512)
    sweep_parser.add_argument("--repeats", type=int, default=3)
    data_parser = commands.add_parser("datasets")
    data_parser.add_argument("--t-end", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "validate":
        validate()
    elif args.command == "sweep":
        sweep(args)
    else:
        datasets(args)


if __name__ == "__main__":
    main()
