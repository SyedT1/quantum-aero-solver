"""Corrected periodic D2Q9 BGK solver and benchmark diagnostics.

The challenge PDF overloads ``L`` as both the box length and the vortex
length.  This module keeps them separate.  The canonical periodic case is
``box_length=2*pi`` and ``vortex_length=1``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np

C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
     [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int8,
)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
CS2 = 1 / 3


@dataclass(frozen=True)
class LBMConfig:
    n: int
    reynolds: float
    box_length: float = 2 * np.pi
    vortex_length: float = 1.0
    vortex_velocity: float = 1.0
    convection_x: float = 1.0
    convection_y: float = 0.0
    density: float = 1.0
    t_end: float = 1.0
    mach: float = 0.05
    snapshots: int = 21

    def validate(self) -> None:
        if self.n < 4 or self.reynolds <= 0 or self.t_end <= 0:
            raise ValueError("n >= 4, reynolds > 0 and t_end > 0 are required")
        if self.vortex_length <= 0 or self.box_length <= 0 or self.mach <= 0:
            raise ValueError("lengths and mach must be positive")
        periods = self.box_length / (2 * np.pi * self.vortex_length)
        if not np.isclose(periods, round(periods), atol=1e-12):
            raise ValueError(
                "Periodic boundaries require box_length/(2*pi*vortex_length) "
                "to be an integer"
            )


def tgv_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    *,
    viscosity: float,
    vortex_length: float = 1.0,
    vortex_velocity: float = 1.0,
    convection_x: float = 1.0,
    convection_y: float = 0.0,
    density: float = 1.0,
    reference_pressure: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact periodic velocity and pressure fields."""
    xi = (x - convection_x * t) / vortex_length
    eta = (y - convection_y * t) / vortex_length
    decay = np.exp(-2 * viscosity * t / vortex_length**2)
    u = convection_x + vortex_velocity * np.sin(xi) * np.cos(eta) * decay
    v = convection_y - vortex_velocity * np.cos(xi) * np.sin(eta) * decay
    pressure = reference_pressure + density * vortex_velocity**2 / 4 * (
        np.cos(2 * xi) + np.cos(2 * eta)
    ) * decay**2
    return u, v, pressure


def equilibrium(rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """D2Q9 second-order equilibrium, populations on the final axis."""
    cu = u[..., None] * C[:, 0] + v[..., None] * C[:, 1]
    usq = u * u + v * v
    return W * rho[..., None] * (
        1 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * usq[..., None] / CS2
    )


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=-1)
    u = (f * C[:, 0]).sum(axis=-1) / rho
    v = (f * C[:, 1]).sum(axis=-1) / rho
    return rho, u, v


def stream(f: np.ndarray) -> np.ndarray:
    out = np.empty_like(f)
    for i, (cx, cy) in enumerate(C):
        out[..., i] = np.roll(np.roll(f[..., i], int(cx), axis=0), int(cy), axis=1)
    return out


def relative_l2(
    u: np.ndarray, v: np.ndarray, ue: np.ndarray, ve: np.ndarray
) -> float:
    numerator = np.mean((u - ue) ** 2 + (v - ve) ** 2)
    denominator = np.mean(ue**2 + ve**2)
    return float(np.sqrt(numerator / denominator))


def periodic_divergence_l2(u: np.ndarray, v: np.ndarray, dx: float) -> float:
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
    return float(np.sqrt(np.mean((dudx + dvdy) ** 2)))


def _diagnostics(
    f: np.ndarray,
    cfg: LBMConfig,
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    velocity_scale: float,
) -> dict[str, float]:
    rho, ul, vl = macroscopic(f)
    u, v = ul / velocity_scale, vl / velocity_scale
    nu = cfg.vortex_velocity * cfg.vortex_length / cfg.reynolds
    ue, ve, _ = tgv_exact(
        x, y, t,
        viscosity=nu,
        vortex_length=cfg.vortex_length,
        vortex_velocity=cfg.vortex_velocity,
        convection_x=cfg.convection_x,
        convection_y=cfg.convection_y,
        density=cfg.density,
    )
    du = u - cfg.convection_x
    dv = v - cfg.convection_y
    due = ue - cfg.convection_x
    dve = ve - cfg.convection_y
    return {
        "relative_l2": relative_l2(u, v, ue, ve),
        "vortex_relative_l2": relative_l2(du, dv, due, dve),
        "kinetic_energy": float(0.5 * np.mean(u * u + v * v)),
        "exact_kinetic_energy": float(0.5 * np.mean(ue * ue + ve * ve)),
        "mass_relative_drift": float(abs(rho.mean() - cfg.density) / cfg.density),
        "divergence_l2": periodic_divergence_l2(u, v, cfg.box_length / cfg.n),
    }


def run_lbm(
    cfg: LBMConfig,
    *,
    collision: Callable[[np.ndarray, float], np.ndarray] | None = None,
    keep_fields: bool = False,
) -> dict:
    """Run BGK LBM and return reproducible timing and fidelity diagnostics.

    A custom collision accepts ``(f, omega)`` and operates on ``(..., 9)``.
    Timing covers time integration only and deliberately excludes setup/output.
    """
    cfg.validate()
    nu = cfg.vortex_velocity * cfg.vortex_length / cfg.reynolds
    dx = cfg.box_length / cfg.n
    max_speed = np.hypot(
        abs(cfg.convection_x) + cfg.vortex_velocity,
        abs(cfg.convection_y) + cfg.vortex_velocity,
    )
    velocity_scale = cfg.mach * np.sqrt(CS2) / max_speed
    dt_nominal = dx * velocity_scale
    nsteps = max(1, int(np.ceil(cfg.t_end / dt_nominal)))
    dt = cfg.t_end / nsteps
    velocity_scale = dt / dx
    nu_lattice = nu * dt / dx**2
    tau = 0.5 + nu_lattice / CS2
    omega = 1 / tau
    if not 0 < omega < 2:
        raise ValueError(f"Unstable BGK relaxation omega={omega:g}")

    coords = (np.arange(cfg.n) + 0.5) * dx
    x, y = np.meshgrid(coords, coords, indexing="ij")
    u0, v0, _ = tgv_exact(
        x, y, 0,
        viscosity=nu,
        vortex_length=cfg.vortex_length,
        vortex_velocity=cfg.vortex_velocity,
        convection_x=cfg.convection_x,
        convection_y=cfg.convection_y,
        density=cfg.density,
    )
    rho = np.full((cfg.n, cfg.n), cfg.density)
    f = equilibrium(rho, u0 * velocity_scale, v0 * velocity_scale)

    wanted = np.unique(np.linspace(0, nsteps, cfg.snapshots, dtype=int))
    wanted_set = set(map(int, wanted))
    records: list[dict] = []
    fields: list[dict] = []

    def record(step: int) -> None:
        diag = _diagnostics(f, cfg, x, y, step * dt, velocity_scale)
        records.append({"step": step, "time": step * dt, **diag})
        if keep_fields:
            r, ul, vl = macroscopic(f)
            up, vp = ul / velocity_scale, vl / velocity_scale
            ue, ve, _ = tgv_exact(
                x, y, step * dt,
                viscosity=nu,
                vortex_length=cfg.vortex_length,
                vortex_velocity=cfg.vortex_velocity,
                convection_x=cfg.convection_x,
                convection_y=cfg.convection_y,
                density=cfg.density,
            )
            fields.append(
                {"f": np.moveaxis(f.copy(), -1, 0), "u": up, "v": vp,
                 "u_exact": ue, "v_exact": ve,
                 "f_eq_exact": np.moveaxis(
                     equilibrium(np.full_like(ue, cfg.density),
                                 ue * velocity_scale, ve * velocity_scale), -1, 0)}
            )

    record(0)
    start = perf_counter()
    for step in range(1, nsteps + 1):
        rho, u, v = macroscopic(f)
        if collision is None:
            f -= omega * (f - equilibrium(rho, u, v))
        else:
            f = collision(f, omega)
        f = stream(f)
        if step in wanted_set:
            record(step)
    runtime = perf_counter() - start

    result = {
        "config": asdict(cfg),
        "convention": "periodic: box_length=2*pi, vortex_length=1",
        "viscosity": nu,
        "dx": dx,
        "dt": dt,
        "velocity_scale": velocity_scale,
        "tau": tau,
        "omega": omega,
        "steps": nsteps,
        "runtime_seconds": runtime,
        "seconds_per_step": runtime / nsteps,
        "population_memory_bytes": f.nbytes,
        "records": records,
    }
    if keep_fields:
        result["fields"] = fields
    return result
