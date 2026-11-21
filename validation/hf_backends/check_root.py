"""Replay the archived C6H12+ force using a transported UHF root and frozen models."""

import argparse
import csv
from pathlib import Path
import runpy

import numpy as np
from pyscf import gto, lib, scf

from deepks.deephf import HFState, build_hf_state, make_deephf
from deepks.model import SpinCorrNet


def apply_smoothed_control(models, archive):
    """Replay the paper's frozen-B transform and read its matching PySCF force."""
    tau = 0.001
    with (archive / "forces.csv").open(newline="") as stream:
        row, = (row for row in csv.DictReader(stream)
                if (row["case"], row["model"], row["point"]) == ("stress", "B", "old_endpoint")
                and float(row["tau"]) == tau)
    # The user supplies the trusted paper archive. Use the same differentiable
    # transform in both the energy-only and analytic-gradient calls.
    features = runpy.run_path(str(archive / "smooth.py"))["features"]

    def smooth_input(_model, args):
        return (features(args[0], tau), *args[1:])

    for model in models:
        model.register_forward_pre_hook(smooth_input)
    return float(row["force_direction"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="archive with z, xyz (Angstrom), d, e and force (eV/Angstrom)")
    parser.add_argument("weights", nargs="+", type=Path, help="joint B ensemble checkpoints")
    parser.add_argument("--engine", choices=["pyscf", "gpu4pyscf", "generativeqc"], default="generativeqc")
    parser.add_argument("--smoothed-control", type=Path,
                        help="trusted paper archive containing smooth.py and its forces.csv")
    args = parser.parse_args()
    with np.load(args.root) as data:
        mol = gto.M(atom=list(zip(data["z"], data["xyz"])), basis="def2-svp",
                    charge=1, spin=1, unit="Angstrom", verbose=0)
        seed = HFState(float(data["e"]), data["d"]).response_reference(scf.UHF(mol).density_fit())
        expected = float(data["force"][4, 0])
    models = [SpinCorrNet.load(path) for path in args.weights]
    if args.smoothed_control is not None:
        expected = apply_smoothed_control(models, args.smoothed_control)
    options = dict(family="uhf", density_fit=True, scf_backend=args.engine, initial_reference=seed)
    if args.engine == "generativeqc":
        options["scf_args"] = dict(conv_tol=1e-13, density_tolerance=1e-12, max_cycle=120)
    hartree_ev = 27.211386245988
    xyz = mol.atom_coords()
    energies = {}
    for h in (0., 3e-6, -3e-6, 1e-6, -1e-6, 3e-7, -3e-7):
        target = xyz.copy()
        target[4, 0] += h / lib.param.BOHR
        new_mol = mol.set_geom_(target, unit="Bohr", inplace=False)
        state = build_hf_state(new_mol, **options)
        ref = state.response_reference(scf.UHF(new_mol).density_fit())
        cross = gto.intor_cross("int1e_ovlp", mol, new_mol)
        overlaps = [
            np.linalg.svd(old[:, occ > 0].T @ cross @ new[:, new_occ > 0], compute_uv=False).min()
            for old, occ, new, new_occ in zip(seed.mo_coeff, seed.mo_occ, ref.mo_coeff, ref.mo_occ)
        ]
        if min(overlaps) < .999:
            raise RuntimeError(f"occupied subspace changed at displacement {h}: {overlaps}")
        print(f"h={h:g} Angstrom  minimum occupied overlap={min(overlaps):.10f}", flush=True)
        correction = np.mean([model.correction_from_density(new_mol, state.density) for model in models])
        energies[h] = np.array([state.energy, correction])
        if h == 0:
            force = np.mean([make_deephf(ref, model).forces()[4, 0] for model in models])
            analytic = force * hartree_ev / lib.param.BOHR
            print(f"analytic force={analytic:.8f}, archive={expected:.8f} eV/Angstrom", flush=True)
            np.testing.assert_allclose(analytic, expected, rtol=0, atol=.005)
        if h < 0:
            finite = (energies[h] - energies[-h]).sum() * hartree_ev / (-2 * h)
            print(f"centered energy difference={finite:.8f} eV/Angstrom", flush=True)
            # At the smallest step SCF density noise is amplified by 1/h.
            # Retain that diagnostic alongside the two resolved step sizes.
            if abs(h) >= 1e-6:
                np.testing.assert_allclose(finite, analytic, rtol=0, atol=.005)


if __name__ == "__main__":
    main()
