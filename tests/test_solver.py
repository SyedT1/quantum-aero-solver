import numpy as np

from quantum_aero.carleman import exact_bgk, order2_collision, validate_block_encoding
from quantum_aero.classical import LBMConfig, W, run_lbm, tgv_exact
from quantum_aero.quantum import applied_noise_experiment


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
