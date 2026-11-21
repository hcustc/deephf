# Molecular HF backend validation

The tests evaluate a frozen DeePHF spin checkpoint on the same neutral RHF
and cation UHF methanol geometries with PySCF, GPU4PySCF and GenerativeQC.
They use spherical def2-SVP, the same JK fitting basis, and PySCF cross
overlap integrals for the projectors. They compare shell descriptors, HF and
corrected energies, all Cartesian forces, and a central difference of the
corrected energy for one force component per state. Further tests exercise
root transport, warm starts after a geometry displacement, CLI configuration,
and GenerativeQC's RHF/UHF Fock action against the direct relaxed Jacobian.

```bash
python -m pytest -q tests/hf_backends
```

With the review fixes on top of `master` at `7e33d2a`, all 1,049 tests
passed on an RTX 5090 with the locked Linux environment (PySCF 2.14.0,
GPU4PySCF 1.8.1, PyTorch 2.11.0+cu128) and the optional GenerativeQC
library. The 55 HF backend tests include two-frame CLI root tracking,
continuation from a GPU-native reference, and negative-spin UHF cold and
warm starts with GenerativeQC on both CPU and CUDA.

The optional-engine tests run when the engine is importable. CUDA tests use
a visible GPU and a CUDA-enabled Torch installation. The original backend comparison
used an RTX 5090, GPU4PySCF 1.7.4, and a GenerativeQC CUDA
build with native source identity
`3b31877336dd65af68a072e89d993f244138aa6e53dc455b8d33b3cb954697cc`.
The following methanol measurements used
PySCF 2.13.1 and the frozen joint-B spin model. The first four columns of
differences give maxima over both electronic states and all components.
The last column gives the signed change in the predicted ionization energy
relative to PySCF.

| Engine vs PySCF | HF energy (Ha) | Corrected energy (Ha) | Descriptor | Force (Ha/Bohr) | IE change (eV) |
|---|---:|---:|---:|---:|---:|
| GPU4PySCF | 9.38e−13 | 1.09e−9 | 5.73e−8 | 5.60e−8 | −7.89e−9 |
| GenerativeQC CUDA | 6.54e−13 | 9.13e−11 | 1.12e−8 | 1.22e−8 | −1.95e−9 |

Before integration with the GPU workflow, the locked Linux CPU suite
had 923 passes, 23 skips and one failure. The base commit `8716f1d` had
916 passes, one skip and the same failure in
`test_spin_resolved_density_response_matches_fresh_uks[0.0001-2e-07]`.
On the local macOS host, both revisions produced the same 49 failures and
206 errors by test ID. The candidate had 669 passes and the base had 662.

The tested GenerativeQC Python FockPlan accepts and returns NumPy arrays.
On CUDA, each adjoint iteration transfers its trial density to host memory
and the returned J/K response to the torch device. The timings below cover energy predictions only.

## Timing complete energy predictions

The table gives seconds per DF-RHF + spin-model energy prediction on
def2-SVP water clusters. The five configurations ran sequentially in the same
Slurm allocation on a shared node with AMD EPYC 9354 CPUs and an RTX 5090.
PySCF used 1, 8 or 16 CPU threads pinned to physical cores. Each GPU engine
used one GPU and one CPU thread. Each configuration started in a fresh process.
For each cluster size, the table gives the median of three calls after one
initial call. Every call started SCF from the default density guess and
included density export, projection and CPU model evaluation.

| Waters | Orbital AOs | PySCF, 1 thread | PySCF, 8 threads | PySCF, 16 threads | GPU4PySCF | GenerativeQC CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0.044 | 0.046 | 0.046 | 0.220 | 0.039 |
| 4 | 96 | 0.327 | 0.148 | 0.126 | 0.315 | 0.220 |
| 8 | 192 | 3.634 | 0.962 | 0.754 | 0.563 | 1.023 |

At 96 AOs, 16-thread PySCF gave the shortest elapsed time. At 192 AOs,
GPU4PySCF was 1.34 times as fast as 16-thread PySCF. For one water, the
first calls in the new processes took 0.049 s (single-thread PySCF),
2.047 s (GPU4PySCF) and 1.734 s (GenerativeQC). Timings for analytic
forces and geometry optimization with a previous density as the SCF guess
require separate measurements.

```bash
for threads in 1 8 16; do
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads \
        python -m validation.hf_backends.benchmark pyscf --threads "$threads"
done
for engine in gpu4pyscf generativeqc; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python -m validation.hf_backends.benchmark "$engine" --threads 1
done
```

## Archived open-shell stress case

`check_root.py` replays the fifth atom's x force at the archived last accepted
geometry of the C6H12 cation (QM9 000658). Supply the replayed canonical UHF
root and the three joint-B checkpoint paths in seed order. The NPZ contains
atomic numbers `z`, Cartesian coordinates `xyz` (Å), alpha/beta AO densities
`d`, the HF energy `e` (Hartree) and the archived ensemble `force` (eV/Å).

```bash
python -m validation.hf_backends.check_root stress-root.npz seed_41/B.pth seed_43/B.pth seed_47/B.pth
```

Add `--smoothed-control /path/to/qm9_ie_spectral_diagnostic` to repeat the
calculation with the paper's inference-time smoothing at τ = 0.001. This
option executes the archive's `smooth.py` and reads the matching PySCF force
from `forces.csv`. Both columns below use the same three original B
checkpoints. The smoothed-B column applies the transform before each
network call, including when differentiating the energy.

The default unseeded GPU4PySCF and GenerativeQC SCF attempts failed to
converge at this geometry. With the archived occupied spaces as a seed,
GenerativeQC converged at the geometry and both sides of each displacement.
The comparison uses the mean corrected energy of the three checkpoints.
The analytic result uses the DeePHF/PySCF Z-vector implementation and the
GenerativeQC-converged state. The finite differences use fresh GenerativeQC
SCF energies and densities at the displaced geometries with the same PySCF
projector integrals and DeePHF model. They bypass the analytic response.

| Force component (eV/Å) | Hard-sorted B | Smoothed B, frozen weights |
|---|---:|---:|
| Archived PySCF analytic | −106.61278336 | −106.59691240 |
| GenerativeQC SCF with the existing analytic response | −106.61278373 | −106.59691276 |
| Centered difference, step 0.000003 Å | −106.61558531 | −106.59961762 |
| Centered difference, step 0.000001 Å | −106.61310835 | −106.59722662 |
| Centered difference, step 0.0000003 Å | −106.60692055 | −106.59105034 |

The smallest occupied-space singular value over these displacements exceeded
0.9999979 in both spins. At the 0.000001 Å step, the absolute differences
from the analytic result were 0.000325 and 0.000314 eV/Å for hard-sorted
and smoothed B. Reducing the step again increased both differences to about
0.00586 eV/Å. This checks the model's total force on the selected root.
The positive soft mode that dominates the orbital force and the distinct
negative mode that establishes internal instability were identified in the
original PySCF orbital-Hessian decomposition.

## Fixed-geometry acetonitrile cation forces

`profile_forces.py` replays the B trajectory in the geometry-optimization
archive. The cation initial event and the next three accepted events are
lines 35, 36, 39 and 41, including the accepted symmetry-breaking seed at
step 0. The initial XYZ supplies C, C, N, H, H, H order. Its provenance
and the displacements between accepted events establish Å as the length unit.
The frozen repository `joint_B.pth` is evaluated on each DF-UHF doublet
using spherical def2-SVP, def2-SVP-JKFIT and explicit `backend="zvector"`.
The archived energies and spin expectation values check the HF branch.
The energy and force comparisons use the same single checkpoint in both runs.

Run from the DeePHF checkout. Set `ARCHIVE` to the directory containing
`initial/` and `runs/`. The commands use one CPU thread. Run each CUDA
command in an existing single-GPU allocation with the corresponding engine.
The model and response contractions use CPU by default in all three cases.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
ARCHIVE=../../outputs/qm9_ie_geometry_opt
python -m validation.hf_backends.profile_forces pyscf "$ARCHIVE" --output pyscf.json
python -m validation.hf_backends.profile_forces generativeqc "$ARCHIVE" --scf-device cpu --output gqc-cpu.json
python -m validation.hf_backends.profile_forces generativeqc "$ARCHIVE" --scf-device cuda --output gqc-cuda.json
python -m validation.hf_backends.profile_forces gpu4pyscf "$ARCHIVE" --output gpu4pyscf.json
```

Each pass creates a new CLI reference sequence. The first frame uses the
engine's independent guess. Later frames use the previous accepted raw AO
density for PySCF/GPU4PySCF or the previous reference with its source
geometry for GenerativeQC. PySCF uses DIIS space 48 and a 300-cycle limit.
With the default eight-vector history, the first native solve missed the
gradient tolerance even after 400 cycles on the local host.

Wall time spans molecule construction, the SCF and root-continuity checks,
corrected energy, analytic gradient, and returned descriptor. Input/output,
model loading, profiling-hook installation and independent post-step
diagnostics are outside the clock. Three plain/profile pairs alternate order
in one process, starting with plain. Every pass resets the electronic seed.
Each profiled interval synchronizes the active CUDA device at both ends when
either the SCF or the response uses CUDA. Plain runs synchronize only at step
boundaries.

The eight exclusive stage times sum to profiled wall time. SCF includes
reference setup, seed handling and density export. External root transport
covers canonicalization and its Hamiltonian checks. Descriptor/network time
includes the energy call, gradient backpropagation, explicit projector motion,
and output descriptor. Orbital adjoint includes DF factor transformation,
GMRES and the metric contraction. The DF Fock derivative includes building
the integral derivatives and contracting the orbital probes. The two
`get_veff` counters are nested subsets of transport and stationarity,
respectively. They exclude SCF iterations and the off-clock residual check.

### Apple M4, one CPU thread

The [PySCF measurements](acetonitrile_pyscf.json) used PySCF 2.14.0 and
PyTorch 2.13.0. The recorded checkout `bfb94e1` adds the CorrNet routing
documentation to `7f7b282`; the numerical code is identical. The table
gives arithmetic means in milliseconds over three initial-frame measurements
and nine continued-frame measurements. Every continued frame uses its actual
preceding accepted geometry. Plain wall times and profiled stages come from
separate passes. Individual wall times, SCF cycles, stage samples, energies,
all force components and root checks are in the JSON.

| PySCF stage | Initial (3) | Continued (9) |
| --- | ---: | ---: |
| Plain whole step | 952.3 | 577.6 |
| Profiled whole step | 764.1 | 607.7 |
| SCF | 423.3 | 236.8 |
| External root transport | 0.0 | 0.0 |
| Reference stationarity | 2.8 | 2.9 |
| Descriptor and network | 6.9 | 7.1 |
| HF gradient | 95.8 | 99.5 |
| Orbital adjoint | 155.6 | 174.9 |
| DF Fock integral derivatives | 77.4 | 83.5 |
| Remaining step work | 2.3 | 2.9 |

The three plain medians per accepted geometry were 712.5, 505.5 and 540.3
ms, following an initial-frame median of 970.7 ms. The four SCF cycle counts
were 100, 76, 36 and 42. All profiled runs reproduced the plain HF energy,
corrected energy and all 18 force components on every frame exactly. The
largest canonical residual was 2.39e−9, and the smallest adjacent occupied
overlap was 0.94055. PySCF made zero transport-site and one validation-site
`get_veff` call per frame. The latter averaged 2.23 ms initially and 2.36 ms
on continuation. The native path supplies no duplicate at these sites.

The optional GenerativeQC CPU build tested on this Mac was
`b1afee7f7bcfcbd984082df4f5d2ad1a878e639c`. Its independent first SCF
reached a stationary root 0.01737306 Ha above the archived root. With
`--seed-initial`, PySCF prepared the first-geometry root once (0.408 s)
outside the replay. GenerativeQC reproduced that root in 8.400 s plain or
8.071 s profiled. On line 36 it did not converge with the requested previous
reference in 300 cycles. The runner retains the matched prefix and both
failure records, then exits nonzero. See the [independent attempt](acetonitrile_gqc_independent.json)
and [anchored attempt](acetonitrile_gqc_anchored.json). CUDA was unavailable on the Mac.

### RTX 5090 server

All configurations below ran serially on node1 during Slurm allocation
19127, on AMD EPYC 9354 cores 28–29 with one compute thread and, where
used, a single RTX 5090 (GPU index 2). The concurrent warm-start job had
separate cores and GPU index 3. PySCF 2.14.0, GPU4PySCF 1.8.1 and
PyTorch 2.11.0+cu128 used the same archive, weights and code as above.
Times are mean milliseconds. The independent first call of each process
includes lazy initialization. Continued times pool nine frame samples
from three replays. Response device controls the spin network and orbital
contractions. The HF nuclear gradient and DF integral derivatives use PySCF on CPU.

| SCF engine | Response device | Plain initial | Plain continued | Profiled continued | Transport get_veff (continued) | Validation get_veff (continued) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [PySCF](acetonitrile_n1/pyscf.json) | CPU | 3083.9 | 1142.7 | 1140.3 | 0 | 5.27 |
| [PySCF](acetonitrile_n1/pyscf_cuda_response.json) | CUDA | 2896.1 | 875.0 | 872.6 | 0 | 5.34 |
| [GPU4PySCF](acetonitrile_n1/gpu4pyscf.json) | CPU | 2878.8 | 1227.6 | 1241.0 | 50.10 | 5.44 |
| [GPU4PySCF](acetonitrile_n1/gpu4pyscf_cuda_response.json) | CUDA | 2645.7 | 944.4 | 950.5 | 52.95 | 5.42 |
| [GenerativeQC CUDA, anchored](acetonitrile_n1/generativeqc_cuda_anchored.json) | CPU | 1374.5 | 1089.2 | 1089.9 | 49.40 | 5.34 |

The [independent GenerativeQC CUDA run](acetonitrile_n1/generativeqc_cuda.json)
selected the same higher-energy first root as the Mac CPU build. The anchored
run reached all four archived geometries. Its first-frame wall time excludes
the one-time same-geometry PySCF anchor preparation recorded in the JSON.
To reproduce, add `--seed-initial` to the GenerativeQC command above.
Use `--device cuda` to select CUDA response contractions in the other rows.
The largest plain/profile changes across the complete server runs were
2.14e−12 Ha in corrected energy and 5.56e−10 Ha/Bohr in any force component.
The JSON files include every exclusive stage for every frame.

Each accepted external frame had one transport and one validation
`get_veff` call on the same reference object. Transport uses the exported
density. Validation rebuilds the density from the canonical orbitals.
Their elements differed by up to 1.4e−13. The opportunity in the measured
continuations is the second call, about 0.44–0.57% of the profiled whole step.
