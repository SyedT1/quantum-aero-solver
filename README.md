# Quantum Aero Solver

Hybrid classical/quantum experiments for the Airbus Global Quantum + AI Challenge 2026, Track A: solve the two-dimensional convecting Taylor-Green vortex and study accuracy, runtime, memory, and quantum-resource scaling as the Reynolds number increases.

The repository currently contains:

- a D2Q9 BGK lattice Boltzmann method (LBM) solver and four generated HDF5 datasets;
- a custom Qiskit circuit for the unitary D2Q9 streaming operation;
- a corrected, periodic Taylor-Green benchmark and Reynolds-number scaling study;
- order-1, order-2, and order-3 Carleman collision experiments;
- a toy Linear Combination of Unitaries (LCU) collision primitive;
- `qlbm` streaming circuits and a simplified composed collision-plus-streaming circuit.

> [!IMPORTANT]
> This is an experimental research repository, not yet a submission-grade end-to-end quantum CFD solver. The most complete notebook is `baseline/Airbus_TrackA_v3_Extended.ipynb`, but its composed collision circuit uses a simplified one-direction-qubit-per-axis LCU primitive rather than the full nine-population BGK/Carleman operator.

## Repository contents

| Path | Purpose | Status |
|---|---|---|
| [`Airbus-Challenge-Statement-vF.pdf`](Airbus-Challenge-Statement-vF.pdf) | Source challenge statement and required scaling metrics | Reference |
| [`baseline/classical_lbm_tgv.ipynb`](baseline/classical_lbm_tgv.ipynb) | Classical D2Q9 solver, validation plots, and HDF5 dataset generation | Executed; generated the committed datasets |
| [`baseline/quantum-streaming-circuit.ipynb`](baseline/quantum-streaming-circuit.ipynb) | Hand-built Qiskit streaming circuit, state encoding, unitarity, scaling, and multi-step checks | Executed at small scale |
| [`baseline/Airbus_TrackA_v2_Scaled.ipynb`](baseline/Airbus_TrackA_v2_Scaled.ipynb) | Corrected periodic TGV baseline, Carleman collision, toy LCU, `qlbm`, and scaling study | Executed; retained as an intermediate version |
| [`baseline/Airbus_TrackA_v3_Extended.ipynb`](baseline/Airbus_TrackA_v3_Extended.ipynb) | v2 plus composed collision/streaming circuits and order-3 Carleman study | Latest experiment notebook |
| [`baseline/dataset/`](baseline/dataset/) | Four gzip-compressed HDF5 simulation datasets | Present via Git LFS |
| [`baseline/dataset.rar`](baseline/dataset.rar) | Archived dataset bundle | Present via Git LFS; contents are not independently documented here |

The notebooks report generated PNGs and serialized circuits, but those artifacts are not currently committed. All numeric results below are taken from the notebooks' stored outputs; runtime values are machine-dependent.

## Mathematical problem

### Incompressible Navier-Stokes equations

For velocity \(\mathbf{u}=(u,v)\), constant density \(\rho\), pressure \(p\), and kinematic viscosity \(\nu\),

$$
\nabla\cdot\mathbf{u}=0,
$$

$$
\frac{\partial\mathbf{u}}{\partial t}
+(\mathbf{u}\cdot\nabla)\mathbf{u}
=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf{u}.
$$

The challenge benchmark is the convecting Taylor-Green vortex. Define

$$
\xi=\frac{x-U_ct}{L_c},\qquad
\eta=\frac{y-V_ct}{L_c},\qquad
A(t)=\exp\!\left(-\frac{2\nu t}{L_c^2}\right).
$$

Its analytical fields are

$$
u(x,y,t)=U_c+V_0\sin(\xi)\cos(\eta)A(t),
$$

$$
v(x,y,t)=V_c-V_0\cos(\xi)\sin(\eta)A(t),
$$

$$
p(x,y,t)=p_0+\frac{\rho V_0^2}{4}
\left[\cos(2\xi)+\cos(2\eta)\right]A(t)^2.
$$

The Reynolds number fixes viscosity through

$$
\mathrm{Re}=\frac{V_0L_c}{\nu},\qquad
\nu=\frac{V_0L_c}{\mathrm{Re}}.
$$

### Length-scale convention

The challenge table gives a domain length of \(2\pi\), while the same symbol appears as the characteristic scale inside the trigonometric arguments. Interpreting both literally on \([0,2\pi)^2\) makes \(\sin(x/2\pi)\) non-periodic at the boundary.

Two conventions therefore exist in this repository:

- `classical_lbm_tgv.ipynb` uses \(L_c=L_{box}=2\pi\). This is the legacy run that produced the committed HDF5 files and its relative L2 error remains approximately 0.25-0.33.
- `Airbus_TrackA_v2_Scaled.ipynb` and v3 separate the scales: \(L_{box}=2\pi\) and \(L_c=1\). This is the canonical periodic TGV convention and reduces the reported error to approximately \(3.6\times10^{-3}\) to \(6.1\times10^{-3}\).

Do not compare the committed dataset errors directly with the corrected v2/v3 results without regenerating the datasets under the same convention.

## D2Q9 lattice Boltzmann formulation

The lattice uses nine discrete velocities

$$
\mathbf{e}_i\in\{(0,0),(1,0),(0,1),(-1,0),(0,-1),
(1,1),(-1,1),(-1,-1),(1,-1)\},
$$

with weights

$$
w_0=\frac49,\qquad
w_{1,2,3,4}=\frac19,\qquad
w_{5,6,7,8}=\frac1{36},\qquad c_s^2=\frac13.
$$

Macroscopic density and lattice velocity are reconstructed from the populations \(f_i\):

$$
\rho=\sum_i f_i,\qquad
\mathbf{u}_{lat}=\frac{1}{\rho}\sum_i f_i\mathbf{e}_i.
$$

The second-order equilibrium distribution is

$$
f_i^{eq}=w_i\rho\left[
1+\frac{\mathbf{e}_i\cdot\mathbf{u}_{lat}}{c_s^2}
+\frac{(\mathbf{e}_i\cdot\mathbf{u}_{lat})^2}{2c_s^4}
-\frac{\mathbf{u}_{lat}\cdot\mathbf{u}_{lat}}{2c_s^2}
\right].
$$

One BGK update comprises collision

$$
f_i^*(\mathbf{x},t)=f_i(\mathbf{x},t)
-\omega\left[f_i(\mathbf{x},t)-f_i^{eq}(\mathbf{x},t)\right],
\qquad \omega=\frac1\tau,
$$

followed by periodic streaming

$$
f_i(\mathbf{x}+\mathbf{e}_i,t+1)=f_i^*(\mathbf{x},t).
$$

The lattice relaxation time and physical/lattice-unit conversion used by the dataset generator are

$$
\tau=\frac12+\frac{\nu_{lat}}{c_s^2},\qquad
\nu_{lat}=\nu\frac{\Delta t}{\Delta x^2},\qquad
\mathbf{u}_{phys}=\mathbf{u}_{lat}\frac{\Delta x}{\Delta t}.
$$

## Validation metrics

The notebooks use relative velocity-field L2 error

$$
\epsilon_{L2}=
\frac{\sqrt{\left\langle
(u-u_{exact})^2+(v-v_{exact})^2
\right\rangle}}
{\sqrt{\left\langle u_{exact}^2+v_{exact}^2\right\rangle}+10^{-12}},
$$

and mean kinetic energy

$$
K(t)=\frac12\left\langle u^2+v^2\right\rangle,
$$

where \(\langle\cdot\rangle\) is the spatial grid mean.

## Experiments and results

### 1. Classical dataset-generation run (legacy length convention)

`classical_lbm_tgv.ipynb` runs to \(t=1\) s and stores 21 snapshots for each Reynolds number.

| Re | Grid | \(\tau\) | Steps | Mach | Final L2 | \(K_{LBM}\) | \(K_{exact}\) | Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 32 x 32 | 0.7400 | 203 | 0.087 | 2.9041e-1 | 1.0281 | 0.9144 | 0.2 s |
| 100 | 64 x 64 | 0.5480 | 407 | 0.087 | 3.3074e-1 | 1.0479 | 0.9385 | 0.5 s |
| 400 | 128 x 128 | 0.5144 | 1,358 | 0.052 | 2.5108e-1 | 1.0671 | 0.9431 | 3.9 s |
| 1,000 | 256 x 256 | 0.5077 | 4,074 | 0.035 | 2.4668e-1 | 1.0520 | 0.9439 | 47.1 s |

These are internally verified stored outputs, but the large, grid-insensitive errors are explained by the length-scale mismatch above. The datasets remain useful for exercising data-loading and quantum-streaming code, but should be regenerated with \(L_c=1\) before being treated as canonical TGV training data.

### 2. Corrected periodic classical baseline

The v2/v3 notebooks use \(L_c=1\) and report this grid-convergence study:

| Re | N=32 | N=64 | N=128 | N=256 |
|---:|---:|---:|---:|---:|
| 10 | 4.8478e-3 | 3.6861e-3 | 3.6054e-3 | 3.6233e-3 |
| 100 | 6.0666e-3 | 4.9574e-3 | 4.8803e-3 | 4.9068e-3 |

The remaining plateau is consistent with the notebook's fixed-Mach, weakly compressible BGK setup; a lower Mach number or a more accurate collision model is needed to push it down.

The separate sandbox scaling run uses \(t_{end}=0.5\) s:

| Re | N | Final L2 | Runtime | Classical population memory | Notebook qubit estimate |
|---:|---:|---:|---:|---:|---:|
| 10 | 32 | 5.1778e-3 | 0.08 s | 0.07 MB | 15 |
| 100 | 64 | 3.9326e-3 | 0.42 s | 0.29 MB | 17 |
| 400 | 128 | 3.7738e-3 | 3.22 s | 1.18 MB | 19 |
| 1,000 | 256 | 3.8465e-3 | 31.70 s | 4.72 MB | 21 |

The last column is an analytical estimate used in the notebook, not an executed circuit measurement. It uses \(2\log_2N+5\), whereas the measured `qlbm` totals below follow \(2\log_2N+7\); this two-qubit discrepancy should be resolved before a final resource comparison.

### 3. Custom quantum streaming circuit

Streaming is a cyclic permutation and therefore unitary. The custom Qiskit implementation amplitude-encodes a \(4\times4\) subset of the Re=10 dataset and applies controlled modular shifts selected by the direction register.

For one streaming step at \(N=4\):

- quantum statevector versus classical streaming: maximum absolute difference \(8.12\times10^{-14}\);
- maximum relative difference: \(1.83\times10^{-13}\);
- unitarity check: \(\max|U^\dagger U-I|=1.96\times10^{-13}\) for the \(256\times256\) operator;
- after five steps, maximum absolute divergence remains \(4.00\times10^{-13}\).

The position and direction state requires

$$
q=2\log_2N+4
$$

qubits in this custom encoding. Stored circuit summaries report 8-18 qubits for \(N=4\) to 128, with depth 29 and 58 high-level gates. These depth and gate counts include undecomposed custom controlled-increment gates, so they are logical-circuit figures, not hardware-native transpiled resource counts.

> [!WARNING]
> The notebook's noise-analysis cell creates a `NoiseModel` but never supplies it to an `AerSimulator` run. It then compares an ideal statevector with the classical result. Consequently, the displayed fidelity of 1.0 at all nominal error rates is not evidence of noise robustness and the associated conclusion should not be used.

### 4. Carleman-linearized collision

With a weakly compressible closure around \(\rho\approx\rho_0\), the equilibrium is represented as a quadratic map

$$
f^{eq}(f)\approx Lf+Q:(f\otimes f).
$$

Writing \(F_1=f\), \(F_2=f\otimes f\), \(R=(1-\omega)I+\omega L\), and \(Q_{eff}=\omega Q\), collision becomes

$$
F_1'=RF_1+Q_{eff}:F_2.
$$

The order-2 closure advances the lifted state approximately as

$$
F_2'\approx(R\otimes R)F_2,
$$

dropping cubic and quartic lifted terms. In 300 random single-collision tests, the mean relative error improves from \(3.111\times10^{-3}\) at order 1 to \(1.566\times10^{-5}\) at order 2. After 200 collision-only steps, stored drift is \(1.999\times10^{-3}\) for order 1 and \(9.994\times10^{-6}\) for order 2.

v3 also retains the cubic cross-term

$$
F_2'=(R\otimes R)F_2+
\left(R\otimes Q_{eff}+Q_{eff}\otimes R\right):F_3,
$$

with \(F_3'\approx(R\otimes R\otimes R)F_3\). For tested perturbation magnitudes 0.03, 0.10, 0.20, and 0.30 over 50 steps, order 3 produced the same drift as order 2: 4.1035e-5, 4.5896e-4, 1.8712e-3, and 4.3037e-3, respectively. The added \(F_2\) correction had norm 0.0508445, but its contraction into the next \(F_1\) update was only \(7.12\times10^{-19}\). This is a null result for the specific closure and tests used here, not a general claim that third-order Carleman truncation never helps.

### 5. Toy LCU collision primitive

Because BGK collision is dissipative and non-unitary, v2/v3 test a two-qubit ancilla-and-post-selection circuit as an LCU-style building block. For \(\omega=1.2\), the notebook sets

$$
\theta=\frac{\pi}{2}\min\!\left(\frac{\omega}{2},1\right)=\frac{3\pi}{10}
$$

and applies a controlled \(R_y(2\theta)=R_y(3\pi/5)\). In a 20,000-shot Aer run, the recorded post-selection success probability was 0.794. This circuit demonstrates the ancilla mechanism only: it is not a block encoding of the full D2Q9 collision matrix, and its success probability has not been included in the end-to-end cost estimates.

### 6. `qlbm` streaming resources

The v2/v3 notebooks build `qlbm` collisionless Quantum Transport Method circuits. Stored high-level circuit measurements are:

| N | Total qubits | Grid qubits | Depth | Gates |
|---:|---:|---:|---:|---:|
| 8 | 13 | 6 | 19 | 48 |
| 16 | 15 | 8 | 25 | 72 |
| 32 | 17 | 10 | 31 | 96 |
| 64 | 19 | 12 | 37 | 128 |
| 128 | 21 | 14 | 43 | 160 |

The measured qubit relation is \(q=2\log_2N+7\) for this configuration. As above, these are library-level circuit metrics before a specified hardware target, transpilation basis, routing, and error-correction model.

### 7. Simplified composed collision plus streaming circuit

v3 alternates a toy LCU collision on the two velocity-direction qubits with the real `qlbm` streaming circuit. It adds two reusable ancillas. A small Aer run at \(N=8\), one timestep, and 500 shots completed in 7.4 s and returned four distinct bitstrings. This establishes executability, not physical agreement with a full BGK update.

| N | Timesteps | Qubits | Depth | Gates |
|---:|---:|---:|---:|---:|
| 8 | 1 / 3 / 5 | 15 | 20 / 58 / 96 | 62 / 174 / 286 |
| 16 | 1 / 3 / 5 | 17 | 26 / 76 / 126 | 88 / 248 / 408 |
| 32 | 1 / 3 / 5 | 19 | 32 / 94 / 156 | 114 / 322 / 530 |
| 64 | 1 / 3 / 5 | 21 | 38 / 112 / 186 | 148 / 420 / 692 |

The observed high-level qubit count is \(2\log_2N+9\), while depth and gate count increase approximately linearly with the composed timestep count in this small study.

## Datasets

All files contain 21 snapshots and are gzip-compressed HDF5. Sizes below are current on-disk decimal sizes.

| File | Re | N | `f`, `f_eq_exact` shape | `u`, `v`, exact-field shape | Size |
|---|---:|---:|---|---|---:|
| `lbm_Re10_N32.h5` | 10 | 32 | `(21, 9, 32, 32)` | `(21, 32, 32)` | 3.39 MB |
| `lbm_Re100_N64.h5` | 100 | 64 | `(21, 9, 64, 64)` | `(21, 64, 64)` | 13.29 MB |
| `lbm_Re400_N128.h5` | 400 | 128 | `(21, 9, 128, 128)` | `(21, 128, 128)` | 51.48 MB |
| `lbm_Re1000_N256.h5` | 1,000 | 256 | `(21, 9, 256, 256)` | `(21, 256, 256)` | 198.59 MB |

Each file has the following schema:

| HDF5 path | Shape | Units/use |
|---|---|---|
| `/metadata` | group attributes | `Re`, `N`, `nu`, `dt`, `dx`, `tau`, `scale`, `L`, `V0`, `Uc`, `Vc`, `T_end` |
| `/times` | `(21,)` | Snapshot time |
| `/u`, `/v` | `(21, N, N)` | LBM velocity fields in physical units; proposed ML/FNO inputs |
| `/f` | `(21, 9, N, N)` | Distribution populations in lattice units; proposed VQC inputs |
| `/u_exact`, `/v_exact` | `(21, N, N)` | Analytical velocity targets in physical units |
| `/f_eq_exact` | `(21, 9, N, N)` | Equilibrium-population collision targets in lattice units |
| `/l2_errors` | `(21,)` | Relative L2 error history |
| `/ke_sim`, `/ke_exact` | `(21,)` | Simulated and analytical mean kinetic energy |

The binary datasets and archive are tracked through Git LFS. After cloning, run `git lfs pull` if they appear as pointer files.

## Running the notebooks

No lockfile or packaged environment is committed. The imports used across the notebooks are:

```text
numpy pandas matplotlib tqdm h5py
qiskit qiskit-aer qlbm
jupyter
```

The stored custom-streaming run installed Qiskit 2.4.2 and Qiskit Aer 0.17.2; versions for the remaining packages were not recorded. For reproducibility, create an isolated environment and pin a working set after the first successful run.

Run Jupyter from `baseline/`, because the notebooks use relative paths such as `dataset/...`:

```powershell
cd baseline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install jupyter numpy pandas matplotlib tqdm h5py qiskit qiskit-aer qlbm
jupyter lab
```

Suggested order:

1. `classical_lbm_tgv.ipynb` only if you intend to regenerate the legacy datasets. For canonical periodic data, first port the v3 \(L_c=1\) convention into this generator.
2. `quantum-streaming-circuit.ipynb` for the hand-built streaming validation.
3. `Airbus_TrackA_v2_Scaled.ipynb` for the corrected baseline and initial resource studies.
4. `Airbus_TrackA_v3_Extended.ipynb` for the latest composed-circuit and order-3 experiments.

The full Re=1,000 legacy dataset-generation run took 47.1 s in its recorded environment; corrected scaling at \(N=256\), \(t_{end}=0.5\), took 31.7 s. Large statevector simulations can require substantially more memory than the compact logical qubit count suggests.

## Current limitations and next work

- Regenerate all HDF5 datasets with the corrected periodic \(L_c=1\) formulation and record provenance in file attributes.
- Replace the direction-bit toy LCU with a block-encoded nine-population Carleman/BGK collision operator.
- Validate the composed quantum timestep against the classical LBM state, not only circuit executability.
- Fix the noise experiment by transpiling to an explicit basis and actually passing the noise model to Aer.
- Report post-transpilation gate counts, depth, connectivity overhead, measurement cost, and success/post-selection probability.
- Resolve the \(2\log_2N+5\) versus \(2\log_2N+7\) qubit-estimate inconsistency.
- Pin dependencies and record hardware, seeds, wall-clock protocol, and repeated timing statistics.
- Add a spectral or high-order finite-volume reference before claiming comparison with a state-of-the-art classical solver.
- Run the proposed production sweep (up to Re=5,000) on suitable compute; the notebook's 10-11 hour figure is an extrapolation, not a completed experiment.
- Add the proposed VQC collision and FNO corrector notebooks; the current repository contains datasets intended for them but no trained VQC/FNO models.

## Project status

Completed here: classical D2Q9 solver, four datasets, exact custom streaming validation, corrected periodic baseline, small-scale Reynolds scaling, Carleman collision studies, `qlbm` streaming resources, and a runnable simplified composed circuit.

Still open: a physically complete nine-population quantum collision, end-to-end quantum/classical accuracy validation, valid noisy-hardware analysis, corrected dataset regeneration, production-scale runs, and the final technical report.
