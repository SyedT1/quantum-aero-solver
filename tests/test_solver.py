import numpy as np

from quantum_aero.carleman import exact_bgk, order2_collision, validate_block_encoding
from quantum_aero.classical import LBMConfig, W, run_lbm, tgv_exact
from quantum_aero.quantum import applied_noise_experiment
from quantum_aero.deliverables import (
    coherent_lifted_trajectory,
    initial_lattice_state,
    run_pseudospectral_tgv,
    sparse_collision_oracle,
)
from quantum_aero.advanced import (
    amplitude_amplification_experiment,
    collision_stability,
    simulate_local_collision,
)


def test_nonperiodic_pdf_literal_is_rejected():
    config = LBMConfig(n=16, reynolds=100, vortex_length=2 * np.pi)
    try:
        config.validate()
    except ValueError as error:
        assert "Periodic boundaries" in str(error)
    else:
        raise AssertionError("the non-periodic literal PDF convention was accepted")


def test_exact_solution_is_periodic_and_divergence_free():
    points = np.linspace(0, 2 * np.pi, 33)
    x, y = np.meshgrid(points, points, indexing="ij")
    u, v, p = tgv_exact(x, y, 0.37, viscosity=0.01)
    assert np.allclose(u[0], u[-1])
    assert np.allclose(v[:, 0], v[:, -1])
    assert np.allclose(p[0], p[-1])


def test_corrected_lbm_has_small_error():
    result = run_lbm(LBMConfig(n=32, reynolds=10, t_end=0.1, snapshots=2))
    assert result["records"][-1]["relative_l2"] < 0.02
    assert result["records"][-1]["mass_relative_drift"] < 1e-12


def test_order2_collision_matches_bgk_near_reference_density():
    velocity = np.array([0.02, -0.01])
    directions = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
         [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    cu = directions @ velocity
    f = W * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * velocity @ velocity)
    assert np.linalg.norm(order2_collision(f, 1.2) - exact_bgk(f, 1.2)) < 1e-12


def test_full_block_encoding_is_unitary_and_correct():
    info = validate_block_encoding(W.copy(), omega=1.2)
    assert info["lifted_dimension"] == 90
    assert info["qubits"] == 8
    assert info["max_block_error"] < 1e-12
    assert info["max_unitarity_error"] < 1e-10


def test_noise_is_applied_and_changes_the_distribution():
    result = applied_noise_experiment(
        shots=4_000, one_qubit_error=0.01, two_qubit_error=0.05, seed=11
    )
    assert result["noise_model_applied"] is True
    assert result["total_variation_distance"] > 0.001


def test_coherent_lifted_recurrence_runs_without_relift():
    cfg = LBMConfig(n=4, reynolds=100, t_end=0.05, snapshots=2)
    f, omega, _, _ = initial_lattice_state(cfg)
    records = coherent_lifted_trajectory(f, omega, checkpoints=(1, 2))
    assert [record["step"] for record in records] == [1, 2]
    assert records[0]["coherent_vs_bgk"] < 1e-12
    assert records[1]["lift_consistency_defect"] > 0


def test_pseudospectral_comparator_matches_exact_tgv():
    result = run_pseudospectral_tgv(
        LBMConfig(n=16, reynolds=100, t_end=0.05, snapshots=2)
    )
    assert result["relative_l2"] < 1e-6


def test_factorized_collision_matches_dense_matrix():
    result = sparse_collision_oracle(omega=1.2)
    assert result["factorized_matvec_max_error"] < 1e-12
    assert result["flat_to_factorized_storage_ratio"] > 10


def test_actual_local_collision_circuit_and_amplification():
    cfg = LBMConfig(n=4, reynolds=100, t_end=0.05, snapshots=2)
    f, omega, _, _ = initial_lattice_state(cfg)
    result = simulate_local_collision(f[0, 0], omega, steps=2)
    assert result["conditional_fidelity"] > 1 - 1e-12
    amplified = amplitude_amplification_experiment(f[0, 0], omega, max_iterations=12)
    assert max(row["success_probability"] for row in amplified) > 0.95


def test_off_equilibrium_collision_stability_probe():
    result = collision_stability(omega=1.95, speed=0.1, density_shift=0.02, steps=200)
    assert result["stable"] is True
    assert np.isfinite(result["relative_error"])
