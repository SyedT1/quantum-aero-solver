"""Reusable kernels for the six proposal-strengthening experiment notebooks.

The functions in this module favor small, auditable experiments.  In
particular, ``coherent_lifted_trajectory`` evolves both levels of the global
order-2 Carleman state without rebuilding the quadratic level from the
first-level state between timesteps.  It is a linear-algebra emulator of the
coherent recurrence, not a claim that a fault-tolerant circuit already exists.
"""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .carleman import exact_bgk, lifted_matrix, operators, order2_collision
from .classical import C, CS2, LBMConfig, W, equilibrium, macroscopic, stream, tgv_exact


def initial_lattice_state(cfg: LBMConfig) -> tuple[np.ndarray, float, float, float]:
    """Return ``(f, omega, velocity_scale, dt)`` for the canonical TGV initial state."""
    cfg.validate()
    viscosity = cfg.vortex_velocity * cfg.vortex_length / cfg.reynolds
    dx = cfg.box_length / cfg.n
    max_speed = np.hypot(
        abs(cfg.convection_x) + cfg.vortex_velocity,
        abs(cfg.convection_y) + cfg.vortex_velocity,
    )
    nominal_scale = cfg.mach * np.sqrt(CS2) / max_speed
    steps = max(1, int(np.ceil(cfg.t_end / (dx * nominal_scale))))
    dt = cfg.t_end / steps
    velocity_scale = dt / dx
    tau = 0.5 + viscosity * dt / (dx * dx * CS2)
    omega = 1.0 / tau
    coords = (np.arange(cfg.n) + 0.5) * dx
    x, y = np.meshgrid(coords, coords, indexing="ij")
    u, v, _ = tgv_exact(
        x,
        y,
        0.0,
        viscosity=viscosity,
        vortex_length=cfg.vortex_length,
        vortex_velocity=cfg.vortex_velocity,
        convection_x=cfg.convection_x,
        convection_y=cfg.convection_y,
        density=cfg.density,
    )
    rho = np.full((cfg.n, cfg.n), cfg.density)
    return equilibrium(rho, u * velocity_scale, v * velocity_scale), omega, velocity_scale, dt


def streaming_permutation(n: int) -> np.ndarray:
    """Dense permutation for one D2Q9 streaming step on a small ``n`` by ``n`` grid."""
    dimension = 9 * n * n
    permutation = np.zeros((dimension, dimension))
    for x in range(n):
        for y in range(n):
            for direction, (cx, cy) in enumerate(C):
                source = (x * n + y) * 9 + direction
                destination = (((x + int(cx)) % n) * n + (y + int(cy)) % n) * 9 + direction
                permutation[destination, source] = 1.0
    return permutation


def global_linear_map(n: int, omega: float) -> np.ndarray:
    """Linear part of collision followed by streaming."""
    linear, _ = operators()
    local_r = (1 - omega) * np.eye(9) + omega * linear
    return streaming_permutation(n) @ np.kron(np.eye(n * n), local_r)


def _quadratic_action(level_two: np.ndarray, n: int, omega: float) -> np.ndarray:
    """Apply the local quadratic collision term to a global level-two state."""
    _, quadratic = operators()
    result = np.zeros(9 * n * n)
    for site in range(n * n):
        sl = slice(9 * site, 9 * (site + 1))
        result[sl] = omega * np.einsum("ijk,jk->i", quadratic, level_two[sl, sl])
    return streaming_permutation(n) @ result


def vector_exact_step(vector: np.ndarray, n: int, omega: float) -> np.ndarray:
    f = vector.reshape(n, n, 9)
    return stream(exact_bgk(f, omega)).reshape(-1)


def vector_relift_step(vector: np.ndarray, n: int, omega: float) -> np.ndarray:
    f = vector.reshape(n, n, 9)
    return stream(order2_collision(f, omega)).reshape(-1)


def coherent_lifted_trajectory(
    f0: np.ndarray, omega: float, checkpoints: tuple[int, ...] = (1, 2, 5, 10)
) -> list[dict[str, float]]:
    """Compare exact BGK, classical re-lifting, and coherent order-2 evolution.

    The coherent branch advances ``F2`` as ``A F2 A.T`` and never replaces it
    with ``outer(F1, F1)``.  That is the defining test performed here.
    """
    n = f0.shape[0]
    first = f0.reshape(-1).copy()
    second = np.outer(first, first)
    exact = first.copy()
    relift = first.copy()
    a = global_linear_map(n, omega)
    records: list[dict[str, float]] = []
    for step in range(1, max(checkpoints) + 1):
        exact = vector_exact_step(exact, n, omega)
        relift = vector_relift_step(relift, n, omega)
        first_next = a @ first + _quadratic_action(second, n, omega)
        second_next = a @ second @ a.T
        first, second = first_next, second_next
        if step in checkpoints:
            denominator = np.linalg.norm(exact)
            consistency = np.linalg.norm(second - np.outer(first, first)) / np.linalg.norm(second)
            records.append(
                {
                    "step": step,
                    "coherent_vs_bgk": float(np.linalg.norm(first - exact) / denominator),
                    "relift_vs_bgk": float(np.linalg.norm(relift - exact) / denominator),
                    "coherent_vs_relift": float(np.linalg.norm(first - relift) / np.linalg.norm(relift)),
                    "lift_consistency_defect": float(consistency),
                    "first_level_norm": float(np.linalg.norm(first)),
                    "second_level_norm": float(np.linalg.norm(second)),
                }
            )
    return records


def macroscopic_observables(vector: np.ndarray, n: int, velocity_scale: float) -> dict[str, float]:
    rho, ul, vl = macroscopic(vector.reshape(n, n, 9))
    u, v = ul / velocity_scale, vl / velocity_scale
    vortex_u = u - float(u.mean())
    mode = np.fft.fft2(vortex_u)[1 % n, 1 % n] / (n * n)
    return {
        "mass": float(rho.mean()),
        "kinetic_energy": float(0.5 * np.mean(u * u + v * v)),
        "mode_11_real": float(mode.real),
        "mode_11_imag": float(mode.imag),
    }


def run_pseudospectral_tgv(
    cfg: LBMConfig, *, cfl: float = 0.25, repeats: int = 1
) -> dict[str, float | int | dict]:
    """Solve periodic 2-D vorticity Navier--Stokes with dealiased RK4."""
    n = cfg.n
    length = cfg.box_length
    viscosity = cfg.vortex_velocity * cfg.vortex_length / cfg.reynolds
    x1 = np.arange(n) * length / n
    x, y = np.meshgrid(x1, x1, indexing="ij")
    u0, v0, _ = tgv_exact(
        x, y, 0.0, viscosity=viscosity, vortex_length=cfg.vortex_length,
        vortex_velocity=cfg.vortex_velocity, convection_x=cfg.convection_x,
        convection_y=cfg.convection_y, density=cfg.density,
    )
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    inv_k2 = np.zeros_like(k2)
    inv_k2[k2 > 0] = 1.0 / k2[k2 > 0]
    cutoff = (2.0 / 3.0) * np.max(np.abs(k))
    dealias = (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)
    omega0 = np.fft.ifft2(1j * kx * np.fft.fft2(v0) - 1j * ky * np.fft.fft2(u0)).real
    omega_hat0 = np.fft.fft2(omega0)
    dt_nominal = cfl * (length / n) / (abs(cfg.convection_x) + cfg.vortex_velocity + 1e-15)
    steps = max(1, int(np.ceil(cfg.t_end / dt_nominal)))
    dt = cfg.t_end / steps

    def velocity(omega_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_hat = omega_hat * inv_k2
        u = cfg.convection_x + np.fft.ifft2(1j * ky * psi_hat).real
        v = cfg.convection_y - np.fft.ifft2(1j * kx * psi_hat).real
        return u, v

    def rhs(omega_hat: np.ndarray) -> np.ndarray:
        u, v = velocity(omega_hat)
        ox = np.fft.ifft2(1j * kx * omega_hat).real
        oy = np.fft.ifft2(1j * ky * omega_hat).real
        nonlinear = np.fft.fft2(-(u * ox + v * oy)) * dealias
        return nonlinear - viscosity * k2 * omega_hat

    timings = []
    final_hat = omega_hat0
    for _ in range(repeats):
        omega_hat = omega_hat0.copy()
        start = perf_counter()
        for _step in range(steps):
            k1 = rhs(omega_hat)
            k2r = rhs(omega_hat + 0.5 * dt * k1)
            k3 = rhs(omega_hat + 0.5 * dt * k2r)
            k4 = rhs(omega_hat + dt * k3)
            omega_hat += dt * (k1 + 2 * k2r + 2 * k3 + k4) / 6
        timings.append(perf_counter() - start)
        final_hat = omega_hat
    u, v = velocity(final_hat)
    ue, ve, _ = tgv_exact(
        x, y, cfg.t_end, viscosity=viscosity, vortex_length=cfg.vortex_length,
        vortex_velocity=cfg.vortex_velocity, convection_x=cfg.convection_x,
        convection_y=cfg.convection_y, density=cfg.density,
    )
    error = np.sqrt(np.mean((u - ue) ** 2 + (v - ve) ** 2) / np.mean(ue**2 + ve**2))
    return {
        "config": asdict(cfg),
        "steps": steps,
        "runtime_median_seconds": float(np.median(timings)),
        "runtime_min_seconds": float(np.min(timings)),
        "runtime_max_seconds": float(np.max(timings)),
        "relative_l2": float(error),
        "memory_bytes": int(8 * n * n * 12),
    }


def postselection_statistics(f: np.ndarray, omega: float) -> dict[str, float]:
    """State-dependent success statistics for every cell in a spatial field."""
    matrix = lifted_matrix(omega)
    alpha = max(float(np.linalg.norm(matrix, 2)), 1.0)
    flat = f.reshape(-1, 9)
    probabilities = []
    for cell in flat:
        z = np.concatenate([cell, np.outer(cell, cell).reshape(-1)])
        probabilities.append(float(np.linalg.norm(matrix @ z) ** 2 / (alpha * alpha * np.linalg.norm(z) ** 2)))
    p = np.asarray(probabilities)
    return {
        "omega": float(omega),
        "alpha": alpha,
        "p_min": float(p.min()),
        "p_median": float(np.median(p)),
        "p_max": float(p.max()),
        "raw_attempts_worst": float(1 / p.min()),
        "aa_scale_worst": float(1 / np.sqrt(p.min())),
    }


def sparse_collision_oracle(omega: float = 1.2, tolerance: float = 1e-14) -> dict:
    """Validate flat-sparse and block/Kronecker collision representations.

    The flat matrix turns out to be nearly dense.  The useful structured
    representation stores the 9x9 ``R`` and 9x9x9 ``Q`` factors and applies
    the lower block as ``R @ F2 @ R.T`` without materializing ``kron(R, R)``.
    """
    matrix = lifted_matrix(omega)
    rows, columns = np.nonzero(np.abs(matrix) > tolerance)
    values = matrix[rows, columns]
    order = np.lexsort((columns, rows))
    rows, columns, values = rows[order], columns[order], values[order]
    row_start = np.searchsorted(rows, np.arange(matrix.shape[0] + 1))

    def oracle_matvec(vector: np.ndarray) -> np.ndarray:
        out = np.zeros_like(vector)
        for row in range(matrix.shape[0]):
            sl = slice(row_start[row], row_start[row + 1])
            out[row] = np.dot(values[sl], vector[columns[sl]])
        return out

    rng = np.random.default_rng(7)
    probe = rng.normal(size=matrix.shape[1])
    error = np.max(np.abs(oracle_matvec(probe) - matrix @ probe))
    linear, quadratic = operators()
    r = (1 - omega) * np.eye(9) + omega * linear
    q = omega * quadratic

    def factorized_matvec(vector: np.ndarray) -> np.ndarray:
        first = vector[:9]
        second = vector[9:].reshape(9, 9)
        first_out = r @ first + np.einsum("ijk,jk->i", q, second)
        second_out = r @ second @ r.T
        return np.concatenate([first_out, second_out.reshape(-1)])

    factorized_error = np.max(np.abs(factorized_matvec(probe) - matrix @ probe))
    row_sparsity = np.diff(row_start)
    unique_magnitudes = np.unique(np.round(np.abs(values), 14))
    factorized_coefficients = int(np.count_nonzero(np.abs(r) > tolerance) + np.count_nonzero(np.abs(q) > tolerance))
    return {
        "dimension": int(matrix.shape[0]),
        "nnz": int(len(values)),
        "density": float(len(values) / matrix.size),
        "max_row_sparsity": int(row_sparsity.max()),
        "median_row_sparsity": float(np.median(row_sparsity)),
        "unique_coefficient_magnitudes": int(len(unique_magnitudes)),
        "oracle_matvec_max_error": float(error),
        "factorized_matvec_max_error": float(factorized_error),
        "r_nnz": int(np.count_nonzero(np.abs(r) > tolerance)),
        "q_nnz": int(np.count_nonzero(np.abs(q) > tolerance)),
        "factorized_stored_coefficients": factorized_coefficients,
        "flat_to_factorized_storage_ratio": float(len(values) / factorized_coefficients),
        "column_index_bits": int(np.ceil(np.log2(matrix.shape[1]))),
        "row_count_bits": int(np.ceil(np.log2(row_sparsity.max() + 1))),
    }
