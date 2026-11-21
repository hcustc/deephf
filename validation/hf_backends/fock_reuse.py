"""Paired trial of one-shot transport Fock reuse on a fixed GPU4PySCF trajectory."""

import argparse
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
from pyscf import lib
import torch

from deepks.deephf.backends.hf import HFState
from deepks.deephf.df_spin import method
from deepks.model import SpinCorrNet
from . import profile_forces as profile


class OneFrameTrial:
    """A single-use witness for the validation experiment's fixed DF-UHF Hamiltonian.

    Only the two targeted calls see overrides. The benchmark independently
    computes the full canonical residual with the ordinary PySCF Fock off-clock.
    No state is attached to a returned SCF object or used across geometries.
    """

    def __init__(self):
        self.witness = None
        self.hits = 0
        self.bounds = []

    @contextmanager
    def enabled(self):
        transport = HFState.response_reference
        validate = method._validate_reference

        def capture(state, ref):
            self.witness = None
            fresh = ref.get_veff

            def build(mol, dm):
                veff = fresh(mol, dm)
                factors = ref.with_df._cderi
                # The proof below covers real in-memory, packed three-centre factors.
                if isinstance(factors, np.ndarray) and factors.dtype == np.float64 and factors.ndim == 2:
                    self.witness = (ref, mol, dm.copy(), veff.copy(), factors.copy())
                return veff

            try:
                with patch.object(ref, "get_veff", build):
                    return transport(state, ref)
            except Exception:
                self.witness = None
                raise

        def check(ref):
            witness, self.witness = self.witness, None
            if witness is None or ref is not witness[0]:
                return validate(ref)
            _, source_mol, source_dm, veff, factors = witness
            fresh = ref.get_veff

            def reuse(mol, dm):
                if (mol is not source_mol or mol.omega or dm.shape != source_dm.shape
                        or not np.array_equal(ref.with_df._cderi, factors)):
                    return fresh(mol, dm)
                # For V_s=J[D_a+D_b]-K[D_s], both linear maps have Frobenius
                # operator norm <= sum_P ||B_P||_F^2. Twice the packed norm
                # bounds that sum. Multiply by max column norm of C to bound
                # the change in any component of F C - S C eps.
                delta = np.asarray(dm) - source_dm
                strength = 2 * np.vdot(factors, factors)
                change = np.linalg.norm(delta.sum(0)) + max(np.linalg.norm(d) for d in delta)
                bound = strength * change * np.linalg.norm(ref.mo_coeff, axis=1).max()
                residual = ((ref.get_hcore() + veff) @ ref.mo_coeff
                            - (ref.get_ovlp() @ ref.mo_coeff) * ref.mo_energy[:, None, :])
                # Leave a numerical margin below the production limit of 1e-7.
                # Borderline cases and corrupted orbitals take the ordinary path.
                if np.max(np.abs(residual)) + bound < 9e-8:
                    self.hits += 1
                    self.bounds.append(float(bound))
                    return veff
                return fresh(mol, dm)

            with patch.object(ref, "get_veff", reuse):
                return validate(ref)

        with patch.object(HFState, "response_reference", capture), patch.object(method, "_validate_reference", check):
            try:
                yield self
            finally:
                self.witness = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("pairs must be positive")
    args.engine, args.scf_device, args.device = "gpu4pyscf", "cuda", "cuda"
    lib.num_threads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    symbols, frames, digest = profile.read_frames(args.archive)
    model = SpinCorrNet.load(profile.CHECKPOINT).eval()
    controls = {"max_cycle": 300, "conv_tol": 1e-12, "diis_space": 48}
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=profile.ROOT, text=True).strip(),
        "trial_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_sha256": sha256(Path(profile.__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": sha256(profile.CHECKPOINT.read_bytes()).hexdigest(),
        "trajectory_sha256": digest, "scf": args.engine, "response": args.device,
        "gpu": torch.cuda.get_device_name(), "backend": "zvector", "threads": 1, "pairs": [],
    }
    # Pair zero warms the software and CUDA kernels. Each run, including
    # subsequent timed runs, starts frame zero from an independent SCF guess.
    for pair in range(args.pairs + 1):
        runs = {}
        for use_witness in ((False, True) if pair % 2 else (True, False)):
            trial = OneFrameTrial()
            with trial.enabled() if use_witness else nullcontext():
                rows, failure = profile.run(args, model, symbols, frames, controls, False,
                                            torch.cuda.synchronize, None)
            if failure:
                raise RuntimeError(f"failed archived frame: {failure}")
            runs["reuse" if use_witness else "plain"] = rows
            if use_witness:
                hits, bounds = trial.hits, trial.bounds
        if hits != len(frames):
            raise RuntimeError(f"reuse guard admitted {hits} of {len(frames)} frames")
        delta = {}
        for key, tol in (("e_hf_Eh", 1e-9), ("e_total_Eh", 1e-9), ("force_Eh_per_Bohr", 1e-7)):
            values = [np.asarray([r[key] for r in runs[s]]) for s in ("plain", "reuse")]
            np.testing.assert_allclose(*values, rtol=0, atol=tol, equal_nan=False)
            delta[key] = float(np.max(np.abs(values[0] - values[1])))
        if pair:
            report["pairs"].append({
                "order": list(runs), "max_abs_delta": delta, "delta_FC_bound": bounds,
                **{s: [{k: r[k] for k in ("line", "wall_s", "canonical_residual")} for r in rows]
                   for s, rows in runs.items()},
            })
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
