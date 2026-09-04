"""HDF5 output with provenance and atomic replacement."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import h5py
import numpy as np

from .classical import LBMConfig, run_lbm


def write_dataset(path: str | Path, cfg: LBMConfig) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = run_lbm(cfg, keep_fields=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    records = result["records"]
    fields = result.pop("fields")
    with h5py.File(temp, "w") as handle:
        metadata = handle.create_group("metadata")
        for key, value in result["config"].items():
            metadata.attrs[key] = value
        # Compatibility aliases used by the original streaming notebook.
        aliases = {
            "Re": cfg.reynolds,
            "N": cfg.n,
            "nu": result["viscosity"],
            "dt": result["dt"],
            "dx": result["dx"],
            "tau": result["tau"],
            "scale": result["velocity_scale"],
            "L": cfg.box_length,
            "Lc": cfg.vortex_length,
            "V0": cfg.vortex_velocity,
            "Uc": cfg.convection_x,
            "Vc": cfg.convection_y,
            "T_end": cfg.t_end,
        }
        for key, value in aliases.items():
            metadata.attrs[key] = value
        metadata.attrs["formula_convention"] = result["convention"]
        metadata.attrs["generator"] = "quantum_aero.io.write_dataset"
        metadata.attrs["python_version"] = platform.python_version()
        metadata.attrs["numpy_version"] = np.__version__
        metadata.attrs["result_summary_json"] = json.dumps(
            {k: v for k, v in result.items() if k not in {"config", "records"}}
        )
        handle.create_dataset("times", data=[r["time"] for r in records])
        for key in ("u", "v", "u_exact", "v_exact", "f", "f_eq_exact"):
            handle.create_dataset(
                key, data=np.stack([frame[key] for frame in fields]),
                compression="gzip", compression_opts=4,
            )
        handle.create_dataset("l2_errors", data=[r["relative_l2"] for r in records])
        handle.create_dataset(
            "vortex_l2_errors", data=[r["vortex_relative_l2"] for r in records]
        )
        handle.create_dataset("ke_sim", data=[r["kinetic_energy"] for r in records])
        handle.create_dataset(
            "ke_exact", data=[r["exact_kinetic_energy"] for r in records]
        )
        handle.create_dataset(
            "divergence_l2", data=[r["divergence_l2"] for r in records]
        )
        handle.flush()
    os.replace(temp, path)
    return result
