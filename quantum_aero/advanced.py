"""Small-circuit and extended-diagnostic experiments for the research gates."""

from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np

from .carleman import lift, lifted_matrix, operators, unitary_dilation
from .classical import C, CS2, LBMConfig, macroscopic, run_lbm, tgv_exact


def local_collision_circuit(f: np.ndarray, omega: float, steps: int = 1):
    """Construct a genuine high-level circuit for repeated local block encodings."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, UnitaryGate

    matrix = lifted_matrix(omega)
    unitary, alpha, padded = unitary_dilation(matrix)
    state = np.zeros(padded, dtype=complex)
    z = lift(f)
    state[: len(z)] = z / np.linalg.norm(z)
    system_qubits = int(np.log2(padded))
    circuit = QuantumCircuit(system_qubits + steps)
    circuit.append(StatePreparation(state), range(system_qubits))
    collision = UnitaryGate(unitary, label="U_M")
    for step in range(steps):
        circuit.append(collision, list(range(system_qubits)) + [system_qubits + step])
    return circuit, matrix, alpha, state


def simulate_local_collision(f: np.ndarray, omega: float, steps: int) -> dict[str, float | int]:
    from qiskit.quantum_info import Statevector

    circuit, matrix, alpha, state = local_collision_circuit(f, omega, steps)
    actual = Statevector.from_instruction(circuit).data[: len(state)]
    expected = state.copy()
    padded_matrix = np.zeros((len(state), len(state)))
    padded_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix / alpha
    for _ in range(steps):
        expected = padded_matrix @ expected
    probability = float(np.vdot(actual, actual).real)
    fidelity = float(abs(np.vdot(actual, expected)) ** 2 / (probability * np.vdot(expected, expected).real))
    return {
        "steps": steps,
        "qubits": circuit.num_qubits,
        "high_level_depth": circuit.depth(),
        "success_probability": probability,
        "conditional_fidelity": fidelity,
        "expected_raw_attempts": float(1 / probability),
    }


def amplitude_amplification_experiment(f: np.ndarray, omega: float, max_iterations: int = 12) -> list[dict]:
    """Apply exact Grover iterates for the state-preparation + block-encoding circuit."""
    from qiskit.quantum_info import Operator

    circuit, _, _, _ = local_collision_circuit(f, omega, steps=1)
    w = Operator(circuit).data
    dimension = w.shape[0]
    half = dimension // 2
    s_good = np.eye(dimension, dtype=complex)
    s_good[:half, :half] *= -1
    s_zero = np.eye(dimension, dtype=complex)
    s_zero[0, 0] = -1
    iterate = -w @ s_zero @ w.conj().T @ s_good
    state = w[:, 0]
    records = []
    for k in range(max_iterations + 1):
        probability = float(np.vdot(state[:half], state[:half]).real)
        records.append({"iterations": k, "success_probability": probability, "block_calls": 2 * k + 1})
        state = iterate @ state
    return records


def increment_gate(n_qubits: int):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(n_qubits, name="INC")
    for index in range(n_qubits - 1, 0, -1):
        circuit.mcx(list(range(index)), index)
    circuit.x(0)
    return circuit.to_gate()


def d2q9_streaming_circuit(n: int):
    """Direction-controlled periodic streaming with explicit high-level shifts."""
    from qiskit import QuantumCircuit

    if n < 2 or n & (n - 1):
        raise ValueError("n must be a power of two")
    position_qubits = int(np.log2(n))
    direction_qubits = 4
    circuit = QuantumCircuit(direction_qubits + 2 * position_qubits)
    directions = list(range(direction_qubits))
    x_register = list(range(direction_qubits, direction_qubits + position_qubits))
    y_register = list(range(direction_qubits + position_qubits, circuit.num_qubits))
    for direction, (cx, cy) in enumerate(C):
        zero_controls = [bit for bit in range(4) if not ((direction >> bit) & 1)]
        for bit in zero_controls:
            circuit.x(bit)
        for shift, register in ((int(cx), x_register), (int(cy), y_register)):
            if shift:
                gate = increment_gate(position_qubits)
                if shift < 0:
                    gate = gate.inverse()
                circuit.append(gate.control(4), directions + register)
        for bit in zero_controls:
            circuit.x(bit)
    return circuit


def compile_streaming_resources(n: int, topology: str = "line") -> dict:
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    circuit = d2q9_streaming_circuit(n)
    if topology == "line":
        coupling = CouplingMap.from_line(circuit.num_qubits, bidirectional=True)
    elif topology == "ring":
        coupling = CouplingMap.from_ring(circuit.num_qubits, bidirectional=True)
    elif topology == "all_to_all":
        coupling = None
    else:
        raise ValueError(topology)
    start = perf_counter()
    compiled = transpile(
        circuit, basis_gates=["u", "cx"], coupling_map=coupling,
        optimization_level=1, seed_transpiler=7,
    )
    operations = dict(Counter(compiled.count_ops()))
    rotations = operations.get("u", 0)
    # Ross--Selinger-style logarithmic synthesis proxy, explicitly labeled.
    t_per_rotation = int(np.ceil(3 * np.log2(1 / 1e-10) + 10))
    return {
        "n": n, "topology": topology, "logical_qubits": circuit.num_qubits,
        "high_level_depth": circuit.depth(), "high_level_size": circuit.size(),
        "native_depth": compiled.depth(), "native_size": compiled.size(),
        "cx": operations.get("cx", 0), "u": rotations,
        "clifford_t_proxy": rotations * t_per_rotation,
        "compile_seconds": perf_counter() - start,
        "unused_direction_states": 7,
    }


def raw_global_state(f: np.ndarray) -> tuple[np.ndarray, float]:
    """Encode raw positive populations in |direction,x,y> Qiskit ordering."""
    n = f.shape[0]
    position_qubits = int(np.log2(n))
    state = np.zeros(2 ** (4 + 2 * position_qubits), dtype=complex)
    for x in range(n):
        for y in range(n):
            for direction in range(9):
                index = direction + (x << 4) + (y << (4 + position_qubits))
                state[index] = f[x, y, direction]
    norm = float(np.linalg.norm(state))
    return state / norm, norm


def decode_raw_shots(counts: dict[int, int], norm: float, n: int) -> np.ndarray:
    """Positive-amplitude tomography estimator from computational-basis shots."""
    total = sum(counts.values())
    position_qubits = int(np.log2(n))
    f = np.zeros((n, n, 9))
    for index, count in counts.items():
        direction = index & 15
        x = (index >> 4) & (n - 1)
        y = (index >> (4 + position_qubits)) & (n - 1)
        if direction < 9:
            f[x, y, direction] = norm * np.sqrt(count / total)
    return f


def raw_state_observable_shots(f: np.ndarray, shots: int, seed: int = 7) -> dict:
    state, norm = raw_global_state(f)
    probabilities = np.abs(state) ** 2
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(shots, probabilities)
    counts = {index: int(value) for index, value in enumerate(samples) if value}
    estimate = decode_raw_shots(counts, norm, f.shape[0])
    rho, u, v = macroscopic(f)
    rho_e, ue, ve = macroscopic(estimate)
    valid = rho_e > 0
    velocity_error = np.sqrt(np.mean((u[valid] - ue[valid]) ** 2 + (v[valid] - ve[valid]) ** 2))
    kinetic = 0.5 * np.mean(u * u + v * v)
    kinetic_e = 0.5 * np.mean(ue[valid] * ue[valid] + ve[valid] * ve[valid])
    mode = np.fft.fft2(u - u.mean())[1, 1]
    mode_e = np.fft.fft2(ue - ue.mean())[1, 1]
    return {
        "shots": shots, "state_qubits": int(np.log2(len(state))), "normalization": norm,
        "velocity_rmse_lattice": float(velocity_error),
        "kinetic_energy_absolute_error": float(abs(kinetic_e - kinetic)),
        "mode_11_absolute_error": float(abs(mode_e - mode)),
        "unobserved_cells": int(np.size(valid) - np.count_nonzero(valid)),
    }


def extended_lbm_diagnostics(cfg: LBMConfig) -> dict:
    """Run LBM and add pressure, momentum, phase, Fourier, and positivity metrics."""
    result = run_lbm(cfg, keep_fields=True)
    initial = result["fields"][0]
    final = result["fields"][-1]
    f = np.moveaxis(final["f"], 0, -1)
    rho, _, _ = macroscopic(f)
    velocity_scale = result["velocity_scale"]
    pressure = CS2 * (rho - rho.mean()) / velocity_scale**2
    coords = (np.arange(cfg.n) + 0.5) * cfg.box_length / cfg.n
    x, y = np.meshgrid(coords, coords, indexing="ij")
    viscosity = cfg.vortex_velocity * cfg.vortex_length / cfg.reynolds
    _, _, pressure_exact = tgv_exact(
        x, y, cfg.t_end, viscosity=viscosity,
        vortex_length=cfg.vortex_length, vortex_velocity=cfg.vortex_velocity,
        convection_x=cfg.convection_x, convection_y=cfg.convection_y,
        density=cfg.density,
    )
    pressure_exact -= pressure_exact.mean()
    pressure_error = np.linalg.norm(pressure - pressure_exact) / np.linalg.norm(pressure_exact)
    initial_momentum = np.array([initial["u"].mean(), initial["v"].mean()])
    final_momentum = np.array([final["u"].mean(), final["v"].mean()])
    mode = np.fft.fft2(final["u"] - cfg.convection_x)[1, 1]
    mode_exact = np.fft.fft2(final["u_exact"] - cfg.convection_x)[1, 1]
    phase_error = np.angle(np.exp(1j * (np.angle(mode) - np.angle(mode_exact))))
    summary = dict(result["records"][-1])
    summary.update({
        "pressure_relative_l2": float(pressure_error),
        "momentum_drift": float(np.linalg.norm(final_momentum - initial_momentum)),
        "fourier_mode_relative_error": float(abs(mode - mode_exact) / abs(mode_exact)),
        "fourier_phase_error_radians": float(abs(phase_error)),
        "minimum_population": float(f.min()),
        "negative_population_count": int(np.count_nonzero(f < 0)),
        "runtime_seconds": result["runtime_seconds"], "steps": result["steps"],
        "tau": result["tau"], "memory_bytes": result["population_memory_bytes"],
    })
    return summary


def collision_stability(omega: float, speed: float, density_shift: float, steps: int) -> dict:
    """Collision-only order-1/order-2 coherent stability on off-equilibrium states."""
    linear, quadratic = operators()
    r = (1 - omega) * np.eye(9) + omega * linear
    q = omega * quadratic
    velocity = np.array([speed, -0.7 * speed])
    cu = C @ velocity
    f0 = W_like = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
    f0 = W_like * (1 + density_shift) * (1 + cu / CS2 + 0.5 * (cu / CS2) ** 2 - 0.5 * velocity @ velocity / CS2)
    exact = f0.copy()
    first = f0.copy()
    second = np.outer(f0, f0)
    max_norm = np.linalg.norm(first)
    for _ in range(steps):
        rho = exact.sum(); u = (exact @ C[:, 0]) / rho; v = (exact @ C[:, 1]) / rho
        cue = C[:, 0] * u + C[:, 1] * v
        feq = W_like * rho * (1 + cue / CS2 + 0.5 * (cue / CS2) ** 2 - 0.5 * (u*u+v*v) / CS2)
        exact = (1 - omega) * exact + omega * feq
        first, second = r @ first + np.einsum("ijk,jk->i", q, second), r @ second @ r.T
        max_norm = max(max_norm, np.linalg.norm(first))
        if not np.all(np.isfinite(first)) or max_norm > 1e8:
            break
    return {
        "omega": omega, "speed": speed, "density_shift": density_shift, "requested_steps": steps,
        "stable": bool(np.all(np.isfinite(first)) and max_norm <= 1e8),
        "relative_error": float(np.linalg.norm(first - exact) / np.linalg.norm(exact)) if np.all(np.isfinite(first)) else float("inf"),
        "minimum_population": float(first.min()) if np.all(np.isfinite(first)) else float("nan"),
        "max_state_norm": float(max_norm),
    }
