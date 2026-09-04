"""Executable noisy streaming test and honest transpiled resource reports."""

from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np


def controlled_periodic_streaming_circuit(measure: bool = True):
    """A controlled x -> x+1 mod 4 streaming primitive."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(3, 3 if measure else 0)
    direction, x0, x1 = 0, 1, 2
    # Non-uniform input: |direction=+>|x=01>.  Preparing every qubit in |+>
    # would produce a uniform distribution that depolarizing noise cannot change.
    circuit.h(direction)
    circuit.x(x0)
    circuit.ccx(direction, x0, x1)
    circuit.cx(direction, x0)
    if measure:
        circuit.measure(range(3), range(3))
    return circuit


def _probabilities(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def hellinger_fidelity(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(np.sqrt(a.get(k, 0) * b.get(k, 0)) for k in keys) ** 2)


def applied_noise_experiment(
    *, shots: int = 20_000, one_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-2, seed: int = 7,
) -> dict:
    """Run the same transpiled circuit on ideal and genuinely noisy Aer backends."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    circuit = controlled_periodic_streaming_circuit(measure=True)
    ideal_backend = AerSimulator()
    transpiled = transpile(
        circuit, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=seed
    )
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(one_qubit_error, 1), ["u"])
    noise.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ["cx"])
    noisy_backend = AerSimulator(noise_model=noise)
    ideal_counts = ideal_backend.run(
        transpiled, shots=shots, seed_simulator=seed
    ).result().get_counts()
    noisy_counts = noisy_backend.run(
        transpiled, shots=shots, seed_simulator=seed
    ).result().get_counts()
    ideal_p = _probabilities(ideal_counts)
    noisy_p = _probabilities(noisy_counts)
    total_variation = 0.5 * sum(
        abs(ideal_p.get(k, 0) - noisy_p.get(k, 0)) for k in set(ideal_p) | set(noisy_p)
    )
    return {
        "shots": shots,
        "seed": seed,
        "one_qubit_error": one_qubit_error,
        "two_qubit_error": two_qubit_error,
        "noise_model_applied": noisy_backend.options.noise_model is not None,
        "basis_gates": ["u", "cx"],
        "transpiled_qubits": transpiled.num_qubits,
        "transpiled_depth": transpiled.depth(),
        "transpiled_operations": dict(Counter(transpiled.count_ops())),
        "hellinger_fidelity": hellinger_fidelity(ideal_p, noisy_p),
        "total_variation_distance": float(total_variation),
        "ideal_counts": dict(ideal_counts),
        "noisy_counts": dict(noisy_counts),
    }


def transpiled_streaming_resources() -> dict:
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    circuit = controlled_periodic_streaming_circuit(measure=False)
    compiled = transpile(
        circuit, basis_gates=["u", "cx"],
        coupling_map=CouplingMap.from_ring(circuit.num_qubits, bidirectional=True),
        optimization_level=1, seed_transpiler=7,
    )
    return {
        "logical_qubits": circuit.num_qubits,
        "high_level_depth": circuit.depth(),
        "high_level_operations": dict(circuit.count_ops()),
        "basis_gates": ["u", "cx"],
        "transpiled_depth": compiled.depth(),
        "transpiled_operations": dict(compiled.count_ops()),
        "connectivity": "bidirectional nearest-neighbor ring",
        "scope": "routed native-basis proxy; no error correction",
    }


def transpiled_collision_resources(omega: float = 1.2) -> dict:
    """Synthesize the complete 8-qubit dense collision dilation to u/cx gates.

    This closes the undecomposed-gate accounting gap, but generic dense unitary
    synthesis is an upper-bound implementation, not a scalable FT construction.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.transpiler import CouplingMap

    from .carleman import lifted_matrix, unitary_dilation

    unitary, alpha, padded = unitary_dilation(lifted_matrix(omega))
    circuit = QuantumCircuit(int(np.log2(unitary.shape[0])))
    circuit.append(UnitaryGate(unitary), range(circuit.num_qubits))
    start = perf_counter()
    compiled = transpile(
        circuit, basis_gates=["u", "cx"],
        coupling_map=CouplingMap.from_ring(circuit.num_qubits, bidirectional=True),
        optimization_level=0, seed_transpiler=7,
    )
    return {
        "omega": omega,
        "logical_qubits": circuit.num_qubits,
        "lifted_dimension": 90,
        "padded_system_dimension": padded,
        "normalization_alpha": alpha,
        "basis_gates": ["u", "cx"],
        "connectivity": "bidirectional nearest-neighbor ring",
        "transpiled_depth": compiled.depth(),
        "transpiled_size": compiled.size(),
        "transpiled_operations": dict(compiled.count_ops()),
        "transpile_seconds": perf_counter() - start,
        "scope": (
            "generic dense routed synthesis; includes no rotation "
            "synthesis to Clifford+T, error correction, or amplitude amplification"
        ),
    }
