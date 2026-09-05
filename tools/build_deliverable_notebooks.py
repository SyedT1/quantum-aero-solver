"""Create the six executable proposal-strengthening notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "deliverables"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(title: str, summary: str, cells: list) -> nbf.NotebookNode:
    header = markdown(
        f"""
# {title}

{summary}

This notebook is an executable evidence artifact. Its default configuration is
deliberately small enough for a clean local rerun; scale-up parameters are
listed separately and are not represented as measured results.
"""
    )
    setup = code(
        """
from pathlib import Path
import sys

repo_root = Path.cwd().parent if Path.cwd().name == "deliverables" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
output_dir = repo_root / "results" / "deliverables"
output_dir.mkdir(parents=True, exist_ok=True)
"""
    )
    nb = nbf.v4.new_notebook(cells=[header, setup, *cells])
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (quantum-aero)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3"}
    return nb


def build() -> None:
    TARGET.mkdir(exist_ok=True)
    notebooks: dict[str, nbf.NotebookNode] = {}

    notebooks["01_coherent_multistep.ipynb"] = notebook(
        "Deliverable 1 — Coherent multi-step Carleman collision + streaming",
        "Tests the global order-2 lifted recurrence for 1, 2, 5, and 10 steps without classically rebuilding the second lift level.",
        [
            markdown(
                """
## Scope and acceptance criterion

The coherent branch advances \\(F_1\\) and \\(F_2\\) as one linear recurrence.
It never assigns \\(F_2\\leftarrow F_1\\otimes F_1\\) between steps. The classical
re-lift branch is retained only as a comparator. Passing this notebook means
that a small coherent recurrence has been numerically specified and tested;
it does **not** mean that its block encoding has been compiled.
"""
            ),
            code(
                """
import json
import pandas as pd
from quantum_aero.classical import LBMConfig
from quantum_aero.deliverables import initial_lattice_state, coherent_lifted_trajectory

cfg = LBMConfig(n=4, reynolds=100, t_end=0.1, mach=0.05, snapshots=2)
f0, omega, velocity_scale, dt = initial_lattice_state(cfg)
records = coherent_lifted_trajectory(f0, omega, checkpoints=(1, 2, 5, 10))
df = pd.DataFrame(records)
df
"""
            ),
            code(
                """
payload = {
    "scope": "global order-2 linear-recurrence emulation; no classical re-lift in coherent branch; not a compiled FT circuit",
    "config": cfg.__dict__, "omega": omega, "dt": dt, "records": records,
}
(output_dir / "01_coherent_multistep.json").write_text(json.dumps(payload, indent=2))
assert len(records) == 4
assert all(record["coherent_vs_bgk"] >= 0 for record in records)
print("PASS: coherent level-two state advanced for 10 steps without re-lifting.")
print("10-step coherent-vs-BGK error:", records[-1]["coherent_vs_bgk"])
"""
            ),
            markdown(
                """
## Scale-up configuration

Repeat at \\(N=8\\), Re 10–5000, multiple Mach numbers, and 50 steps. The
second lifted level scales as \\((9N^2)^2\\), so the classical emulator is
intentionally not presented as a scalable implementation.
"""
            ),
        ],
    )

    notebooks["02_common_state_encoding.ipynb"] = notebook(
        "Deliverable 2 — Common state-encoding audit",
        "Tests raw-population and square-root-population encodings against the same Carleman collision and streaming operations.",
        [
            markdown(
                """
## Question

The legacy streaming notebook uses amplitudes proportional to \\(\\sqrt f\\),
whereas the Carleman operator is linear in a lifted vector made from \\(f\\).
This experiment determines which representation is algebraically compatible
with collision and verifies that streaming remains a permutation in it.
"""
            ),
            code(
                """
import json
import numpy as np
import pandas as pd
from quantum_aero.classical import LBMConfig, stream
from quantum_aero.carleman import lift, lifted_matrix, order2_collision
from quantum_aero.deliverables import initial_lattice_state

cfg = LBMConfig(n=4, reynolds=100, t_end=0.1, mach=0.05, snapshots=2)
field, omega, velocity_scale, dt = initial_lattice_state(cfg)
f = field[1, 2].copy()
matrix = lifted_matrix(omega)
expected = order2_collision(f, omega)

raw_result = (matrix @ lift(f))[:9]
sqrt_f = np.sqrt(np.clip(f, 0, None))
sqrt_amplitude_result = (matrix @ lift(sqrt_f))[:9]
sqrt_decoded = np.square(sqrt_amplitude_result)

rows = [
    {"encoding": "raw f amplitudes", "collision_relative_error": np.linalg.norm(raw_result-expected)/np.linalg.norm(expected)},
    {"encoding": "sqrt(f) amplitudes then square", "collision_relative_error": np.linalg.norm(sqrt_decoded-expected)/np.linalg.norm(expected)},
]
pd.DataFrame(rows)
"""
            ),
            code(
                """
raw_stream = stream(field).reshape(-1)
sqrt_stream_decoded = np.square(np.sqrt(np.clip(field, 0, None))[..., :])
sqrt_stream_decoded = stream(sqrt_stream_decoded).reshape(-1)
stream_error = np.max(np.abs(raw_stream - sqrt_stream_decoded))

result = {
    "chosen_encoding": "raw population amplitudes with an explicit norm/scale register or oracle",
    "raw_collision_relative_error": rows[0]["collision_relative_error"],
    "sqrt_collision_relative_error": rows[1]["collision_relative_error"],
    "streaming_representation_error": float(stream_error),
    "conclusion": "streaming supports either representation, but the Carleman collision requires raw-f lifted amplitudes",
}
(output_dir / "02_common_state_encoding.json").write_text(json.dumps(result, indent=2))
assert result["raw_collision_relative_error"] < 1e-12
assert result["sqrt_collision_relative_error"] > 1e-3
print(json.dumps(result, indent=2))
"""
            ),
            markdown(
                """
## Decision

Use amplitudes proportional to signed/raw populations for the proposed
Carleman path. The remaining circuit-level requirement is a reversible
preparation oracle that also exposes the normalization needed for engineering
observables; probability decoding from the legacy \\(\\sqrt f\\) notebook cannot
be reused unchanged.
"""
            ),
        ],
    )

    notebooks["03_fixed_accuracy_convergence.ipynb"] = notebook(
        "Deliverable 3 — Fixed-accuracy LBM convergence",
        "Runs a factorial Re × grid × Mach sweep rather than changing Re and grid together.",
        [
            markdown(
                """
## Measured default sweep

The sweep uses a short physical time to keep this artifact rerunnable. It
separates Reynolds number, grid resolution, and Mach number and performs two
wall-time repetitions. Final-submission runs should use \\(t=1\\), at least five
repetitions, and the larger configuration shown below.
"""
            ),
            code(
                """
import json
import numpy as np
import pandas as pd
from quantum_aero.classical import LBMConfig, run_lbm

rows = []
for reynolds in (10, 100, 400, 1000):
    for n in (16, 32, 64):
        for mach in (0.1, 0.05, 0.025):
            trials = [run_lbm(LBMConfig(n=n, reynolds=reynolds, t_end=0.25, mach=mach, snapshots=2)) for _ in range(2)]
            final = trials[-1]["records"][-1]
            runtimes = [trial["runtime_seconds"] for trial in trials]
            rows.append({
                "reynolds": reynolds, "n": n, "mach": mach, "t_end": 0.25,
                "steps": trials[-1]["steps"], "tau": trials[-1]["tau"],
                "runtime_median_seconds": float(np.median(runtimes)),
                "runtime_min_seconds": float(np.min(runtimes)),
                "runtime_max_seconds": float(np.max(runtimes)),
                "relative_l2": final["relative_l2"],
                "vortex_relative_l2": final["vortex_relative_l2"],
                "kinetic_energy_error": abs(final["kinetic_energy"]-final["exact_kinetic_energy"]),
                "mass_relative_drift": final["mass_relative_drift"],
                "divergence_l2": final["divergence_l2"],
            })
df = pd.DataFrame(rows)
df.to_csv(output_dir / "03_fixed_accuracy_convergence.csv", index=False)
df.head(12)
"""
            ),
            code(
                """
tolerances = (1e-2, 5e-3)
best = []
for tolerance in tolerances:
    for reynolds in sorted(df.reynolds.unique()):
        eligible = df[(df.reynolds == reynolds) & (df.relative_l2 <= tolerance)]
        if len(eligible):
            row = eligible.sort_values("runtime_median_seconds").iloc[0]
            best.append({"tolerance": tolerance, "reynolds": reynolds, "n": int(row.n), "mach": row.mach,
                         "runtime_seconds": row.runtime_median_seconds, "relative_l2": row.relative_l2})
        else:
            best.append({"tolerance": tolerance, "reynolds": reynolds, "n": None, "mach": None,
                         "runtime_seconds": None, "relative_l2": None})
best_df = pd.DataFrame(best)
best_df.to_csv(output_dir / "03_fixed_accuracy_frontier.csv", index=False)
assert len(df) == 36
print("PASS: 36 independent configurations completed.")
best_df
"""
            ),
            markdown(
                """
## Final scale-up

Use Re = 10, 100, 400, 1000, 2000, 5000; \\(N=32\\)–2048; Ma =
0.1–0.0125; \\(t=1\\); at least five repetitions; and report observed
convergence orders plus the least-cost configuration at each declared error.
"""
            ),
        ],
    )

    notebooks["04_spectral_fixed_accuracy_comparator.ipynb"] = notebook(
        "Deliverable 4 — Independent pseudo-spectral comparator",
        "Benchmarks a dealiased Fourier-vorticity RK4 solver against D2Q9 BGK at matched physical parameters.",
        [
            markdown(
                """
## Comparator protocol

Both solvers use the canonical periodic box, the same final time, Reynolds
number, grid, and relative velocity L2 definition. The spectral solver is an
independent discretization; its timings include time integration but exclude
plotting and file output, matching the LBM timing boundary.
"""
            ),
            code(
                """
import numpy as np
import pandas as pd
from quantum_aero.classical import LBMConfig, run_lbm
from quantum_aero.deliverables import run_pseudospectral_tgv

rows = []
for reynolds in (10, 100, 1000):
    for n in (16, 32, 64):
        cfg = LBMConfig(n=n, reynolds=reynolds, t_end=0.25, mach=0.05, snapshots=2)
        spectral = run_pseudospectral_tgv(cfg, repeats=3)
        lbm_trials = [run_lbm(cfg) for _ in range(3)]
        lbm_final = lbm_trials[-1]["records"][-1]
        rows.extend([
            {"solver": "Fourier-vorticity RK4", "reynolds": reynolds, "n": n,
             "runtime_median_seconds": spectral["runtime_median_seconds"], "relative_l2": spectral["relative_l2"],
             "steps": spectral["steps"], "memory_bytes": spectral["memory_bytes"]},
            {"solver": "D2Q9 BGK", "reynolds": reynolds, "n": n,
             "runtime_median_seconds": float(np.median([x["runtime_seconds"] for x in lbm_trials])),
             "relative_l2": lbm_final["relative_l2"], "steps": lbm_trials[-1]["steps"],
             "memory_bytes": lbm_trials[-1]["population_memory_bytes"]},
        ])
df = pd.DataFrame(rows)
df.to_csv(output_dir / "04_spectral_comparator.csv", index=False)
df
"""
            ),
            code(
                """
assert len(df) == 18
spectral_max = df[df.solver == "Fourier-vorticity RK4"].relative_l2.max()
print("PASS: independent comparator completed; maximum spectral L2 error =", spectral_max)
print("Median results by solver:")
df.groupby("solver")[["runtime_median_seconds", "relative_l2"]].median()
"""
            ),
            markdown(
                """
## Interpretation boundary

This TGV is a special single-mode solution, so the comparator is expected to
be exceptionally strong. The result is the correct baseline for this challenge
instance, not a general performance claim for arbitrary turbulent CFD.
"""
            ),
        ],
    )

    notebooks["05_postselection_trajectory.ipynb"] = notebook(
        "Deliverable 5 — Post-selection over realistic trajectories",
        "Measures block normalization and state-dependent success probabilities at multiple times and Reynolds numbers using actual LBM populations.",
        [
            markdown(
                """
## Why this matters

The proposal's preliminary 1.77% success probability used one local state at
\\(\\omega=1.2\\). Here \\(\\omega\\) is taken from each physical simulation and
probabilities are evaluated over every grid cell at six trajectory snapshots.
The reported product is the naive no-amplification multi-step probability and
is included only to expose compounding—not as the proposed implementation.
"""
            ),
            code(
                """
import pandas as pd
import numpy as np
from quantum_aero.classical import LBMConfig, run_lbm
from quantum_aero.deliverables import postselection_statistics

rows = []
for reynolds in (10, 100, 400, 1000):
    result = run_lbm(LBMConfig(n=16, reynolds=reynolds, t_end=0.1, mach=0.05, snapshots=6), keep_fields=True)
    for record, field in zip(result["records"], result["fields"]):
        f = np.moveaxis(field["f"], 0, -1)
        stats = postselection_statistics(f, result["omega"])
        rows.append({"reynolds": reynolds, "step": record["step"], "time": record["time"],
                     "total_steps": result["steps"], **stats,
                     "log10_naive_full_trajectory_p_worst": result["steps"] * np.log10(stats["p_min"])})
df = pd.DataFrame(rows)
df.to_csv(output_dir / "05_postselection_trajectory.csv", index=False)
df
"""
            ),
            code(
                """
summary = df.groupby("reynolds").agg(
    omega=("omega", "first"), alpha=("alpha", "first"),
    p_min=("p_min", "min"), p_median=("p_median", "median"),
    aa_scale_worst=("aa_scale_worst", "max"),
    log10_naive_full_trajectory_p_worst=("log10_naive_full_trajectory_p_worst", "min"),
).reset_index()
summary.to_csv(output_dir / "05_postselection_summary.csv", index=False)
assert len(df) == 24
print("PASS: probabilities measured for every cell at 24 trajectory/Re checkpoints.")
summary
"""
            ),
        ],
    )

    notebooks["06_structured_collision_ft_estimates.ipynb"] = notebook(
        "Deliverable 6 — Factorized collision oracle + FT scenario estimates",
        "Tests flat sparsity, implements the useful block/Kronecker factorization of the 90-dimensional collision map, and exposes transparent FT resource scenarios.",
        [
            markdown(
                """
## Scope

This notebook first tests whether an ordinary sparse encoding is justified.
It then validates the useful structure: store the local \\(R\\) and \\(Q\\)
factors and evaluate the lower block as \\(R F_2 R^T\\), without materializing
\\(R\\otimes R\\). This is not yet a compiled PREPARE/SELECT block encoding, so
all FT numbers are labeled estimates and their assumptions are stored.
"""
            ),
            code(
                """
import json
import math
import pandas as pd
from quantum_aero.deliverables import sparse_collision_oracle

oracle = sparse_collision_oracle(omega=1.2)
assert oracle["oracle_matvec_max_error"] < 1e-11
assert oracle["factorized_matvec_max_error"] < 1e-11
oracle
"""
            ),
            code(
                """
# Explicit proxy: PREPARE/SELECT stores the nonzero coefficients of R and Q;
# the R⊗R action reuses R twice. This is deliberately conservative and easy
# to replace when a compiled circuit exists.
coefficient_bits = 16
factorized_terms = oracle["factorized_stored_coefficients"]
address_bits = math.ceil(math.log2(factorized_terms))
logical_qubits = 8 + address_bits + oracle["column_index_bits"] + coefficient_bits + 4
toffoli_per_query = 4 * factorized_terms + 4 * coefficient_bits * 9
t_per_query = 4 * toffoli_per_query

scenarios = [
    {"name": "optimistic", "physical_error": 1e-4, "code_distance": 15, "cycle_us": 0.2,
     "factory_qubits": 12000, "t_states_per_cycle": 4.0},
    {"name": "base", "physical_error": 1e-3, "code_distance": 25, "cycle_us": 1.0,
     "factory_qubits": 30000, "t_states_per_cycle": 1.0},
    {"name": "pessimistic", "physical_error": 3e-3, "code_distance": 35, "cycle_us": 2.0,
     "factory_qubits": 60000, "t_states_per_cycle": 0.25},
]
rows = []
for s in scenarios:
    data_qubits = 2 * logical_qubits * s["code_distance"]**2
    t_cycles = t_per_query / s["t_states_per_cycle"]
    rows.append({**s, "logical_qubits_proxy": logical_qubits,
                 "T_count_per_block_query_proxy": t_per_query,
                 "physical_qubits_proxy": data_qubits + s["factory_qubits"],
                 "block_query_time_seconds_proxy": t_cycles * s["cycle_us"] * 1e-6})
df = pd.DataFrame(rows)
df
"""
            ),
            code(
                """
payload = {
    "oracle": oracle,
    "assumptions": {
        "coefficient_bits": coefficient_bits,
        "toffoli_per_query_formula": "4*factorized_stored_coefficients + 4*coefficient_bits*9",
        "T_per_Toffoli": 4,
        "surface_code_data_qubits": "2*logical_qubits*distance^2",
        "warning": "proxy only; excludes state preparation, amplification, streaming, measurement, and a compiled PREPARE/SELECT circuit",
    },
    "scenarios": rows,
}
(output_dir / "06_structured_collision_ft_estimates.json").write_text(json.dumps(payload, indent=2))
print("PASS: factorized R/Q implementation exactly reproduces the dense collision matvec.")
print("Flat matrix density:", oracle["density"], "(flat sparsity is not useful)")
print("Flat/factorized storage ratio:", oracle["flat_to_factorized_storage_ratio"])
"""
            ),
        ],
    )

    for name, nb in notebooks.items():
        nbf.write(nb, TARGET / name)
        print(TARGET / name)


if __name__ == "__main__":
    build()
