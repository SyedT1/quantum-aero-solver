"""One-time repair of invalid claims/cells in the exploratory notebooks.

The production implementation lives in ``quantum_aero``. This script keeps the
notebooks useful without silently retaining known-invalid executable cells.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def clear(cell) -> None:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None


def repair_quantum_streaming() -> None:
    path = ROOT / "baseline" / "quantum-streaming-circuit.ipynb"
    book = nbformat.read(path, as_version=4)
    for cell in book.cells:
        if "unique quantum advantage" in cell.source:
            cell.source = cell.source.replace(
                "This confirms zero accumulation of streaming error — a unique quantum advantage.",
                "This validates ideal unitary streaming only; it is not a quantum-advantage result.",
            )
            clear(cell)
        if cell.cell_type == "code" and "def run_noisy_streaming" in cell.source:
            cell.source = """# Corrected: the noise model is passed to Aer after u/cx transpilation.
from pathlib import Path
import sys

repo_root = Path.cwd().parent if Path.cwd().name == "baseline" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from quantum_aero.quantum import applied_noise_experiment

noise_result = applied_noise_experiment(
    shots=20_000, one_qubit_error=1e-3, two_qubit_error=1e-2, seed=7
)
noise_result
"""
            clear(cell)
    nbformat.write(book, path)


def repair_scaled(path: Path, is_v3: bool) -> None:
    book = nbformat.read(path, as_version=4)
    for cell in book.cells:
        if 'the "state-of-the-art classical solver"' in cell.source:
            cell.source = cell.source.replace(
                'the "state-of-the-art classical solver" comparison point',
                "a classical baseline, not a state-of-the-art reference",
            )
        if "order-2 closure is *already exact*" in cell.source:
            cell.source = cell.source.replace(
                "order-2 closure is *already exact* for this closure's purposes",
                "order-2 and order-3 have the same collision-only trajectory in these tests",
            )
        if "| Re=5000 (advantage regime)" in cell.source:
            cell.source = cell.source.replace(
                "| Re=5000 (advantage regime) | 2048 | 5000 | **~10-11 hours** | needs a workstation/cluster, not a shared sandbox |",
                "| Re=5000 production projection | 2048 | 5000 | ~10-11 hours (unverified) | measured N=128 pilot is in ../results; production run remains open |",
            )
        if "### v2" in cell.source and is_v3:
            cell.source = cell.source.replace("### v2", "### v3", 1)
        if "t_end=10.0" in cell.source:
            cell.source = cell.source.replace(
                "t_end=10.0,       # matches the challenge's own KE-decay figure (Sec 6, t up to 10s)",
                "t_end=1.0,        # challenge PDF benchmark figure is at t=1 s",
            )
        if "2*np.log2(N) + 5" in cell.source:
            cell.source = cell.source.replace(
                "2*np.log2(N) + 5", "2*np.log2(N) + 7"
            ).replace(
                "fit from Sec 5's real qlbm measurements",
                "measured qlbm relation; use quantum_aero resources for native-basis counts",
            )
            clear(cell)

    if is_v3:
        for cell in book.cells:
            if cell.cell_type == "markdown" and cell.source.startswith("## 9. Closing the gap"):
                cell.source = """## 9. Superseded toy composition and corrected collision validation

The earlier direction-bit circuit was not a D2Q9 BGK collision, and resetting
an ancilla was not equivalent to LCU post-selection. It is superseded by an
explicit 90-dimensional order-2 Carleman map and exact 8-qubit unitary dilation
in `../quantum_aero/carleman.py`. The cells below validate the complete local
nine-population collision and a full spatial collision-plus-streaming trajectory.

The spatial experiment rebuilds the nonlinear lift classically between steps;
it is a physics/fidelity validation, not an end-to-end FTQC advantage claim.
"""
            elif cell.cell_type == "code" and "def build_composed_lbm_circuit" in cell.source:
                cell.source = """from pathlib import Path
import sys
repo_root = Path.cwd().parent if Path.cwd().name == "baseline" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from quantum_aero.carleman import validate_block_encoding

block_validation = validate_block_encoding(w.copy(), omega=1.2)
block_validation
"""
                clear(cell)
            elif cell.cell_type == "code" and "circ_sim" in cell.source:
                cell.source = """from quantum_aero.carleman import order2_collision
from quantum_aero.classical import LBMConfig, relative_l2, run_lbm

cfg = LBMConfig(n=32, reynolds=100, t_end=0.1, snapshots=6)
bgk_result = run_lbm(cfg, keep_fields=True)
carleman_result = run_lbm(cfg, collision=order2_collision, keep_fields=True)
bgk_final = bgk_result["fields"][-1]
carleman_final = carleman_result["fields"][-1]
print("velocity relative L2, Carleman vs BGK:", relative_l2(
    carleman_final["u"], carleman_final["v"], bgk_final["u"], bgk_final["v"]
))
"""
                clear(cell)
            elif cell.cell_type == "code" and "composed_rows" in cell.source:
                cell.source = """from quantum_aero.quantum import (
    transpiled_collision_resources, transpiled_streaming_resources
)

collision_resources = transpiled_collision_resources(omega=1.2)
streaming_resources = transpiled_streaming_resources()
collision_resources, streaming_resources
"""
                clear(cell)
            elif cell.cell_type == "code" and "fig, axs" in cell.source and "composed_df" in cell.source:
                cell.source = """print("Dense collision u/cx depth:", collision_resources["transpiled_depth"])
print("Dense collision u/cx operations:", collision_resources["transpiled_operations"])
print("Scope:", collision_resources["scope"])
"""
                clear(cell)
            elif (
                cell.cell_type == "markdown"
                and cell.source.startswith("**Honest result: order-2")
                and "**Scope correction:**" not in cell.source
            ):
                cell.source += """

> **Scope correction:** this null result applies only to the collision-only
> closure tested here. It does not establish end-to-end improvement or make
> order 2 exact for spatial LBM. Section 9 now supplies the required spatial
> collision-plus-streaming comparison.
"""
    nbformat.write(book, path)


def main() -> None:
    repair_quantum_streaming()
    repair_scaled(ROOT / "baseline" / "Airbus_TrackA_v2_Scaled.ipynb", False)
    repair_scaled(ROOT / "baseline" / "Airbus_TrackA_v3_Extended.ipynb", True)


if __name__ == "__main__":
    main()
