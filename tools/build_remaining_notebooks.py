"""Build locally feasible notebooks for research gates 1--11."""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "deliverables"


def md(value: str): return nbf.v4.new_markdown_cell(value.strip())
def code(value: str): return nbf.v4.new_code_cell(value.strip())


def nb(title: str, scope: str, cells: list):
    setup = code("""
from pathlib import Path
import sys
repo_root = Path.cwd().parent if Path.cwd().name == "deliverables" else Path.cwd()
if str(repo_root) not in sys.path: sys.path.insert(0, str(repo_root))
output_dir = repo_root / "results" / "deliverables"
output_dir.mkdir(parents=True, exist_ok=True)
""")
    book = nbf.v4.new_notebook(cells=[md(f"# {title}\n\n{scope}"), setup, *cells])
    book.metadata.kernelspec = {"display_name": "Python 3 (quantum-aero)", "language": "python", "name": "python3"}
    return book


def build():
    books = {}
    books["07_actual_circuit_stateprep_amplification.ipynb"] = nb(
        "Deliverables 1, 2 and 4 — Actual local circuit, raw-f extraction, and amplification",
        "Constructs genuine Qiskit circuits for the full 90-dimensional local collision dilation for 1–10 coherent applications. Streaming is identity in this one-cell circuit; nontrivial global collision/streaming compilation remains a declared blocker.",
        [
            code("""
import json, numpy as np, pandas as pd
from quantum_aero.classical import LBMConfig
from quantum_aero.deliverables import initial_lattice_state
from quantum_aero.advanced import simulate_local_collision, amplitude_amplification_experiment, raw_state_observable_shots

field, omega, velocity_scale, dt = initial_lattice_state(LBMConfig(n=4, reynolds=100, t_end=.1, snapshots=2))
cell = field[0, 0]
circuit_rows = [simulate_local_collision(cell, omega, steps) for steps in (1, 2, 5, 10)]
circuit_df = pd.DataFrame(circuit_rows)
circuit_df
"""),
            code("""
aa_df = pd.DataFrame(amplitude_amplification_experiment(cell, omega, max_iterations=12))
best = aa_df.loc[aa_df.success_probability.idxmax()].to_dict()
print("Best exact amplitude-amplification point:", best)
aa_df
"""),
            code("""
shot_df = pd.DataFrame([raw_state_observable_shots(field, shots, seed=7) for shots in (1_000, 10_000, 100_000, 1_000_000)])
shot_df
"""),
            code("""
payload = {"omega": omega, "collision_circuits": circuit_rows, "amplitude_amplification": aa_df.to_dict("records"),
           "raw_f_observable_extraction": shot_df.to_dict("records"),
           "scope_warning": "Actual local Qiskit circuit. Global nontrivial streaming is not composed with the lifted collision."}
(output_dir / "07_actual_circuit_stateprep_amplification.json").write_text(json.dumps(payload, indent=2))
assert min(x["conditional_fidelity"] for x in circuit_rows) > 1-1e-12
assert best["success_probability"] > .95
print("PASS: exact local circuits, raw-f shot scaling, and amplification executed.")
"""),
        ])

    books["08_factorized_collision_compilation.ipynb"] = nb(
        "Deliverable 3 — Factorized collision compilation audit",
        "Compiles the reusable R block, validates the R/Q/Kronecker matvec, and compares it with the full dense dilation. A complete factorized PREPARE/SELECT circuit is still not claimed.",
        [
            code("""
import json, numpy as np, pandas as pd
from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from quantum_aero.carleman import operators, unitary_dilation
from quantum_aero.deliverables import sparse_collision_oracle
from quantum_aero.quantum import transpiled_collision_resources

omega=1.2
oracle=sparse_collision_oracle(omega)
L,Q=operators(); R=(1-omega)*np.eye(9)+omega*L
UR,alpha_r,padded_r=unitary_dilation(R)
qc=QuantumCircuit(int(np.log2(len(UR)))); qc.append(UnitaryGate(UR), range(qc.num_qubits))
compiled=transpile(qc,basis_gates=["u","cx"],optimization_level=0,seed_transpiler=7)
r_resource={"qubits":qc.num_qubits,"alpha":alpha_r,"native_depth":compiled.depth(),"operations":dict(Counter(compiled.count_ops()))}
r_resource, oracle
"""),
            code("""
dense=transpiled_collision_resources(omega)
rotation_t_cost=int(np.ceil(3*np.log2(1/1e-10)+10))
comparison=pd.DataFrame([
 {"implementation":"dense 256x256 dilation","qubits":dense["logical_qubits"],"native_depth":dense["transpiled_depth"],"cx":dense["transpiled_operations"].get("cx",0),"rotation_T_proxy":dense["transpiled_operations"].get("u",0)*rotation_t_cost},
 {"implementation":"compiled reusable R dilation","qubits":r_resource["qubits"],"native_depth":r_resource["native_depth"],"cx":r_resource["operations"].get("cx",0),"rotation_T_proxy":r_resource["operations"].get("u",0)*rotation_t_cost},
])
comparison
"""),
            code("""
payload={"factorized_validation":oracle,"compiled_R_block":r_resource,"dense_reference":dense,"comparison":comparison.to_dict("records"),
"remaining_blocker":"compile Q loading and coherent R/Q composition as PREPARE/SELECT; R-only compilation is not a full collision block encoding"}
(output_dir/"08_factorized_collision_compilation.json").write_text(json.dumps(payload,indent=2))
assert oracle["factorized_matvec_max_error"]<1e-12
print("PASS with boundary: factorized arithmetic validated and R compiled; full PREPARE/SELECT remains open.")
"""),
        ])

    books["09_production_physics_local_budget.ipynb"] = nb(
        "Deliverable 5 — Production physics sweep under local compute budget",
        "Measures t=1, all requested Reynolds and Mach values, five timing repetitions at N=32, plus N-convergence points through N=64. N=128–2048 remain scale-up runs because the existing N=256 point costs about 510 seconds per repetition.",
        [
            code("""
import json, platform, numpy as np, pandas as pd
from quantum_aero.classical import LBMConfig, run_lbm
from quantum_aero.advanced import extended_lbm_diagnostics

rows=[]
for re in (10,100,400,1000,2000,5000):
 for ma in (.1,.05,.025,.0125):
  cfg=LBMConfig(n=32,reynolds=re,t_end=1,mach=ma,snapshots=2)
  diag=extended_lbm_diagnostics(cfg)
  timings=[diag["runtime_seconds"]]+[run_lbm(cfg)["runtime_seconds"] for _ in range(4)]
  rows.append({"reynolds":re,"n":32,"mach":ma,**diag,
               "runtime_median_seconds":float(np.median(timings)),"runtime_min_seconds":min(timings),"runtime_max_seconds":max(timings)})
df=pd.DataFrame(rows); df.to_csv(output_dir/"09_production_physics_n32.csv",index=False)
df[["reynolds","mach","relative_l2","pressure_relative_l2","fourier_mode_relative_error","minimum_population","runtime_median_seconds"]]
"""),
            code("""
conv=[]
for re in (10,100,400,1000,2000,5000):
 for n in (16,32,64):
  d=extended_lbm_diagnostics(LBMConfig(n=n,reynolds=re,t_end=1,mach=.025,snapshots=2))
  conv.append({"reynolds":re,"n":n,**d})
conv_df=pd.DataFrame(conv)
orders=[]
for re,g in conv_df.groupby("reynolds"):
 g=g.sort_values("n"); e=g.relative_l2.to_numpy(); ns=g.n.to_numpy()
 orders.append({"reynolds":re,"order_16_32":np.log(e[0]/e[1])/np.log(2),"order_32_64":np.log(e[1]/e[2])/np.log(2)})
order_df=pd.DataFrame(orders)
conv_df.to_csv(output_dir/"09_production_physics_convergence.csv",index=False); order_df.to_csv(output_dir/"09_convergence_orders.csv",index=False)
order_df
"""),
            code("""
meta={"python":platform.python_version(),"platform":platform.platform(),"processor":platform.processor(),"measured_rows":len(df)+len(conv_df),
"unmeasured_grids":[128,256,512,1024,2048],"reason":"local time/memory budget; existing N=256 t=1 measurement is ~510 s for one repetition"}
(output_dir/"09_production_physics_metadata.json").write_text(json.dumps(meta,indent=2))
assert len(df)==24 and len(conv_df)==18
print("PASS: requested Re/Ma/t/repetition sweep completed at N=32; resolution extension measured through N=64.")
"""),
        ])

    books["10_stability_noise_streaming_compilation.ipynb"] = nb(
        "Deliverables 7–9 — Stability, noise boundary, and streaming compilation",
        "Runs 50–1000-step off-equilibrium stability tests and native compilation across grid sizes/topologies. The full combined-circuit noise simulation is recorded as dependency-blocked because global lifted collision and streaming are not yet one compiled circuit.",
        [
            code("""
import json, pandas as pd
from quantum_aero.advanced import collision_stability, compile_streaming_resources

stability=[]
for omega in (1.2,1.8,1.95,1.995):
 for speed in (.03,.1,.2,.3):
  for density in (-.02,0,.02):
   for steps in (50,200,1000): stability.append(collision_stability(omega,speed,density,steps))
stability_df=pd.DataFrame(stability); stability_df.to_csv(output_dir/"10_carleman_stability.csv",index=False)
stability_df.groupby(["omega","requested_steps"]).agg(stable_fraction=("stable","mean"),max_error=("relative_error","max"),min_population=("minimum_population","min"))
"""),
            code("""
resources=[]
for n in (4,8,16,32,64,128):
 for topology in ("all_to_all","ring","line"):
  resources.append(compile_streaming_resources(n,topology))
resource_df=pd.DataFrame(resources); resource_df.to_csv(output_dir/"10_streaming_compilation.csv",index=False)
resource_df
"""),
            code("""
noise_status={"status":"blocked","reason":"a nontrivial global raw-f lifted collision+streaming circuit does not yet exist; applying noise to disconnected proxies would not answer the experiment",
"available_evidence":"07 executes the complete local collision circuit; 10 compiles complete streaming separately",
"next_action":"complete Q PREPARE/SELECT and global position-aware lift, then transpile and run Aer noise on that single circuit"}
(output_dir/"10_combined_noise_status.json").write_text(json.dumps(noise_status,indent=2))
assert len(stability_df)==144 and len(resource_df)==18
print("PASS: stability and streaming compilation completed. Combined noise honestly blocked by circuit integration.")
"""),
        ])

    books["11_classical_ft_crossover.ipynb"] = nb(
        "Deliverables 6, 10 and 11 — Classical comparator, hardware-specific FT scenarios, and crossover",
        "Measures the independent spectral comparator with warmups/repetitions and combines measured classical costs with labeled FT proxies. This is a no-go/crossover sensitivity experiment, not a quantum-advantage claim.",
        [
            code("""
import json, platform, numpy as np, pandas as pd
from quantum_aero.classical import LBMConfig, run_lbm
from quantum_aero.deliverables import run_pseudospectral_tgv

classical=[]
for re in (10,100,400,1000,2000,5000):
 for n in (32,64):
  cfg=LBMConfig(n=n,reynolds=re,t_end=1,mach=.05,snapshots=2)
  run_pseudospectral_tgv(cfg,repeats=1); run_lbm(cfg) # warmups
  sp=run_pseudospectral_tgv(cfg,repeats=5)
  lb=[run_lbm(cfg) for _ in range(5)]; last=lb[-1]["records"][-1]
  classical += [
   {"solver":"spectral","reynolds":re,"n":n,"runtime_median_seconds":sp["runtime_median_seconds"],"runtime_min_seconds":sp["runtime_min_seconds"],"runtime_max_seconds":sp["runtime_max_seconds"],"relative_l2":sp["relative_l2"],"memory_bytes":sp["memory_bytes"]},
   {"solver":"LBM","reynolds":re,"n":n,"runtime_median_seconds":float(np.median([x["runtime_seconds"] for x in lb])),"runtime_min_seconds":min(x["runtime_seconds"] for x in lb),"runtime_max_seconds":max(x["runtime_seconds"] for x in lb),"relative_l2":last["relative_l2"],"memory_bytes":lb[-1]["population_memory_bytes"]}]
classical_df=pd.DataFrame(classical); classical_df.to_csv(output_dir/"11_classical_comparator_t1.csv",index=False)
classical_df
"""),
            code("""
factor=json.loads((output_dir/"06_structured_collision_ft_estimates.json").read_text())
post=pd.read_csv(output_dir/"05_postselection_summary.csv")
rows=[]
for scenario in factor["scenarios"]:
 for re in (10,100,400,1000,2000,5000):
  # Use nearest measured post-selection row; 2000/5000 conservatively use Re=1000.
  p=float(post.iloc[(post.reynolds-re).abs().argsort()[:1]].p_min.iloc[0])
  aa=np.ceil(np.pi/(4*np.sqrt(p)))
  classical_best=classical_df[(classical_df.reynolds==re)].runtime_median_seconds.min()
  for nt in (10,100,1000):
   tq=nt*scenario["block_query_time_seconds_proxy"]*(2*aa+1)
   rows.append({"scenario":scenario["name"],"reynolds":re,"timesteps":nt,"p":p,"aa_iterations_proxy":aa,
                "quantum_collision_only_seconds_proxy":tq,"best_measured_classical_seconds":classical_best,
                "ratio_quantum_collision_only_to_classical":tq/classical_best,
                "omitted_quantum_costs":"state preparation, global streaming, extraction, routing between blocks"})
crossover_df=pd.DataFrame(rows); crossover_df.to_csv(output_dir/"11_crossover_no_go.csv",index=False)
crossover_df.groupby(["scenario","timesteps"]).ratio_quantum_collision_only_to_classical.agg(["min","median","max"])
"""),
            code("""
meta={"hardware":platform.platform(),"processor":platform.processor(),"python":platform.python_version(),
"gpu_status":"not measured; no project GPU implementation is present",
"fv_status":"not measured; independent Fourier-vorticity RK4 is the implemented high-order comparator",
"ft_warning":"collision-only lower-bound proxy; absence of global streaming/preparation/extraction makes any favorable ratio insufficient for advantage"}
(output_dir/"11_crossover_metadata.json").write_text(json.dumps(meta,indent=2))
assert len(classical_df)==24 and len(crossover_df)==54
print("PASS: measured classical comparator and FT sensitivity/no-go ledger completed.")
"""),
        ])

    for name,book in books.items():
        nbf.write(book,TARGET/name); print(TARGET/name)


if __name__ == "__main__": build()
