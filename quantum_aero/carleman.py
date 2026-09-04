"""Nine-population order-2 Carleman collision and its block encoding."""

from __future__ import annotations

import numpy as np

from .classical import C, CS2, W, equilibrium, macroscopic


def operators(reference_density: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return L and Q such that f_eq ~= L f + Q:(f tensor f)."""
    ones = np.ones(9)
    directions = C.astype(float)
    linear = np.zeros((9, 9))
    quadratic = np.zeros((9, 9, 9))
    momentum_outer = (
        np.outer(directions[:, 0], directions[:, 0])
        + np.outer(directions[:, 1], directions[:, 1])
    )
    for i in range(9):
        linear[i] = W[i] * ones
        a = (
            directions[i, 0] * directions[:, 0]
            + directions[i, 1] * directions[:, 1]
        ) / (reference_density * CS2)
        linear[i] += W[i] * a
        quadratic[i] = 0.5 * W[i] * np.outer(a, a)
        quadratic[i] -= 0.5 * W[i] * momentum_outer / (reference_density * CS2)
    return linear, quadratic


def exact_bgk(f: np.ndarray, omega: float) -> np.ndarray:
    rho, u, v = macroscopic(f)
    return f - omega * (f - equilibrium(rho, u, v))


def order2_collision(f: np.ndarray, omega: float) -> np.ndarray:
    """Apply the local order-2 closure, rebuilding the lift from classical f."""
    linear, quadratic = operators()
    feq = np.einsum("ij,...j->...i", linear, f)
    feq += np.einsum("ijk,...j,...k->...i", quadratic, f, f)
    return (1 - omega) * f + omega * feq


def lifted_matrix(omega: float) -> np.ndarray:
    """Return the 90x90 local order-2 Carleman collision matrix."""
    linear, quadratic = operators()
    r = (1 - omega) * np.eye(9) + omega * linear
    matrix = np.zeros((90, 90))
    matrix[:9, :9] = r
    matrix[:9, 9:] = (omega * quadratic).reshape(9, 81)
    matrix[9:, 9:] = np.kron(r, r)
    return matrix


def lift(f: np.ndarray) -> np.ndarray:
    if f.shape != (9,):
        raise ValueError("a single nine-population vector is required")
    return np.concatenate([f, np.outer(f, f).reshape(-1)])


def unitary_dilation(
    matrix: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Construct an exact padded unitary dilation of a dense contraction.

    Returns ``(unitary, alpha, padded_dimension)``.  Post-selecting the signal
    ancilla in zero applies ``matrix / alpha`` to the padded input.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("only square matrices are supported")
    padded = 1 << (rows - 1).bit_length()
    alpha = max(float(np.linalg.norm(matrix, 2)), 1.0)
    a = np.zeros((padded, padded), dtype=complex)
    a[:rows, :cols] = matrix / alpha

    def positive_sqrt(value: np.ndarray) -> np.ndarray:
        value = (value + value.conj().T) / 2
        eigvals, eigvecs = np.linalg.eigh(value)
        return (eigvecs * np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.conj().T

    left = positive_sqrt(np.eye(padded) - a @ a.conj().T)
    right = positive_sqrt(np.eye(padded) - a.conj().T @ a)
    unitary = np.block([[a, left], [right, -a.conj().T]])
    return unitary, alpha, padded


def validate_block_encoding(f: np.ndarray, omega: float) -> dict[str, float | int]:
    matrix = lifted_matrix(omega)
    unitary, alpha, padded = unitary_dilation(matrix)
    z = lift(f).astype(complex)
    state = np.zeros(2 * padded, dtype=complex)
    state[: len(z)] = z / np.linalg.norm(z)
    actual = (unitary @ state)[:padded]
    expected = np.zeros(padded, dtype=complex)
    expected[: len(z)] = matrix @ z / (alpha * np.linalg.norm(z))
    probability = float(np.vdot(actual, actual).real)
    unitarity = float(np.max(np.abs(unitary.conj().T @ unitary - np.eye(2 * padded))))
    return {
        "lifted_dimension": len(z),
        "padded_system_dimension": padded,
        "qubits": int(np.log2(2 * padded)),
        "normalization_alpha": alpha,
        "postselection_probability": probability,
        "expected_postselection_attempts": 1 / probability,
        "amplitude_amplification_scale": 1 / np.sqrt(probability),
        "max_block_error": float(np.max(np.abs(actual - expected))),
        "max_unitarity_error": unitarity,
    }
