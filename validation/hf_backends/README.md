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

The optional-engine tests run when the engine is importable. CUDA tests use
a visible GPU and a CUDA-enabled Torch installation. The validation on n1
used an RTX 5090, GPU4PySCF 1.7.4, and a GenerativeQC CUDA
build with native source identity
`3b31877336dd65af68a072e89d993f244138aa6e53dc455b8d33b3cb954697cc`.
The integrated test run with PySCF 2.14 passed all 29 cases, including
conventional CorrNet gradients. The following methanol measurements used
PySCF 2.13.1 and the frozen joint-B spin model. The first four columns of
differences give maxima over both electronic states and all components.
The last column gives the signed change in the predicted ionization energy
relative to PySCF.

| Engine vs PySCF | HF energy (Ha) | Corrected energy (Ha) | Descriptor | Force (Ha/Bohr) | IE change (eV) |
|---|---:|---:|---:|---:|---:|
| GPU4PySCF | 9.38e−13 | 1.09e−9 | 5.73e−8 | 5.60e−8 | −7.89e−9 |
| GenerativeQC CUDA | 6.54e−13 | 9.13e−11 | 1.12e−8 | 1.22e−8 | −1.95e−9 |

In the repository's locked Linux CPU environment the full candidate suite
had 922 passes, 24 skips and one failure. The base commit `8716f1d` had
916 passes, one skip and the same failure in
`test_spin_resolved_density_response_matches_fresh_uks[0.0001-2e-07]`.
On the local macOS host, both revisions produced the same 49 failures and
206 errors by test ID. The candidate had 668 passes and the base had 662.

## Timing complete energy predictions

The following elapsed times are seconds per full DF-RHF + spin-model
energy prediction on small water clusters. Each engine started in a dedicated
process. Each timing includes a new calculation from its default initial
guess, density export, projection and a CPU network evaluation. The table
reports the median of three calls after one untimed-for-the-median call at
that size. Each job used one CPU numerical thread, with one allocated GPU
for the accelerated engines. Other jobs shared the node.

| Waters | Orbital AOs | PySCF | GPU4PySCF | GenerativeQC CUDA |
|---:|---:|---:|---:|---:|
| 1 | 24 | 0.045 | 0.225 | 0.039 |
| 4 | 96 | 0.325 | 0.314 | 0.225 |
| 8 | 192 | 3.638 | 0.588 | 1.034 |

At one water the first call in the fresh process took 0.086, 2.586 and
1.497 s respectively. The ordering of the two GPU engines changed between
four and eight waters in this setup. Other hardware, orbital and auxiliary
bases, thread counts, SCF roots and force calculations need separate timings.

```bash
for engine in pyscf gpu4pyscf generativeqc; do
    python -m validation.hf_backends.benchmark "$engine"
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

The default unseeded GPU4PySCF and GenerativeQC SCF attempts failed to
converge at this geometry. With the archived occupied spaces as a seed,
GenerativeQC converged at the geometry and both sides of each displacement.
The comparison uses the mean corrected energy of the three checkpoints.
The analytic result uses the DeePHF/PySCF Z-vector implementation and the
GenerativeQC-converged state. The finite differences only use GenerativeQC
SCF and the DeePHF energy model.

| Calculation | Force component (eV/Å) |
|---|---:|
| Archived PySCF analytic | −106.61278336 |
| GenerativeQC SCF with the existing analytic response | −106.61278373 |
| Centered difference, step 0.000003 Å | −106.61558531 |
| Centered difference, step 0.000001 Å | −106.61310835 |
| Centered difference, step 0.0000003 Å | −106.60692055 |

The smallest occupied-space singular value over these displacements exceeded
0.9999979 in both spins. The 0.000001 Å step agreed with the analytic force
to 0.00033 eV/Å. Reducing the step again increased the difference to
0.00586 eV/Å as the energy differences approach numerical precision.
This checks the total force on the selected root.
The orbital-Hessian mode decomposition has its original PySCF provenance.
