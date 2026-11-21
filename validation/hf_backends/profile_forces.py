"""Profile the fixed four-frame DF-UHF/Z-vector trajectory through the CLI calculation path."""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np
from pyscf import gto, lib, scf
import torch

from deepks.deephf import workflow
from deepks.deephf.backends.hf import HFState
from deepks.deephf.df_spin import method
from deepks.model import SpinCorrNet


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "tests/spin_energy/data/qm9_methanol/joint_B.pth"
AUXBASIS = "def2-svp-jkfit"
STAGES = ("scf", "transport", "stationarity", "descriptor_network", "hf_gradient",
          "orbital_adjoint", "df_fock_derivative", "other")


def read_frames(archive):
    """Use the ordered nuclei and length unit recorded by the trajectory producer."""
    xyz = archive / "initial/mol_000007.xyz"
    provenance = json.loads(xyz.with_suffix(".provenance.json").read_text())
    count, _, *atoms = xyz.read_text().splitlines()
    symbols = [row.split()[0] for row in atoms]
    if (int(count) != 6 or symbols != list("CCNHHH")
            or provenance["coordinate_units"].lower() != "angstrom"
            or provenance["smiles"] != "CC#N"):
        raise ValueError("expected archived acetonitrile in Angstrom in C C N H H H order")

    path = archive / "runs/mol_000007/B/trajectory.jsonl"
    frames = []
    for line, raw in enumerate(path.read_text().splitlines(), 1):
        record = json.loads(raw)
        if record.get("q") != 1 or record["event"] not in ("initial", "accepted"):
            continue
        if record["event"] != ("accepted" if frames else "initial"):
            raise ValueError(f"unexpected order of cation events at line {line}")
        if np.shape(record["xyz"]) != (len(symbols), 3) or not np.isfinite(record["xyz"]).all():
            raise ValueError(f"invalid coordinates at line {line}")
        if frames:
            displacement = np.subtract(record["xyz"], frames[-1]["xyz"])
            np.testing.assert_allclose(np.linalg.norm(displacement, axis=1).max(),
                                       record["step_A"], atol=1e-12, rtol=0)
        frames.append(dict(line=line, **record))
        if len(frames) == 4:
            return symbols, frames, sha256(path.read_bytes()).hexdigest()
    raise ValueError("expected a cation initial geometry followed by three accepted geometries")


class Timers:
    """Subtract nested stage durations. The two get_veff probes are inclusive subsets."""

    def __init__(self, synchronize):
        self.sync = synchronize
        self.stack = []
        self.seconds = Counter(dict.fromkeys(STAGES, 0.0))
        self.veff = {name: [] for name in ("transport", "stationarity")}
        self.wall = None

    @contextmanager
    def stage(self, name):
        self.stack.append(name)
        self.sync()
        start = perf_counter()
        try:
            yield
        finally:
            self.sync()
            elapsed = perf_counter() - start
            self.stack.pop()
            self.seconds[name] += elapsed
            if self.stack:
                self.seconds[self.stack[-1]] -= elapsed
            else:
                self.wall = elapsed

    def wrap(self, owner, attribute, name):
        original = getattr(owner, attribute)

        @wraps(original)
        def measured(*args, **kwargs):
            with self.stage(name):
                return original(*args, **kwargs)

        return patch.object(owner, attribute, measured)

    @contextmanager
    def instrument(self):
        cls = method.DFSpinDeePHF
        # The gradient's own time covers the descriptor VJP and explicit
        # projector motion. Its HF, orbital and integral children are subtracted.
        targets = (
            (workflow, "build_spin_reference", "scf"),
            (HFState, "response_reference", "transport"),
            (method, "_validate_reference", "stationarity"),
            (SpinCorrNet, "correction_energy", "descriptor_network"),
            (cls, "_gradient_on_validated_root", "descriptor_network"),
            (cls, "descriptor", "descriptor_network"),
            (method, "_hf_gradient", "hf_gradient"),
            (method.DFOrbitalResponse, "__init__", "orbital_adjoint"),
            (cls, "_orbital_gradient", "orbital_adjoint"),
            (method.DFFockDerivative, "__init__", "df_fock_derivative"),
            (method.DFFockDerivative, "contract", "df_fock_derivative"),
        )
        # PySCF 2.14 DF-UHF overrides get_jk and inherits UHF.get_veff.
        original = scf.uhf.UHF.get_veff

        @wraps(original)
        def get_veff(ref, mol=None, dm=None, *args, **kwargs):
            site = self.stack[-1] if self.stack else None
            if site not in self.veff:
                return original(ref, mol, dm, *args, **kwargs)
            self.sync()
            start = perf_counter()
            result = original(ref, mol, dm, *args, **kwargs)
            self.sync()
            # Keep inputs alive until the two call sites can be compared off-clock.
            self.veff[site].append((perf_counter() - start, ref, dm))
            return result

        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(self.wrap(*target))
            stack.enter_context(patch.object(scf.uhf.UHF, "get_veff", get_veff))
            yield

    def details(self):
        sites = {k: {"calls": len(v), "seconds": sum(call[0] for call in v)}
                 for k, v in self.veff.items()}
        transport, validation = self.veff.values()
        match = None
        if len(transport) == len(validation) == 1:
            _, first, d1 = transport[0]
            _, second, d2 = validation[0]
            match = {"same_reference": first is second,
                     "max_abs_density_difference": float(np.max(np.abs(d1 - d2)))}
        return {"exclusive_s": dict(self.seconds), "get_veff": sites, "paired_inputs": match}


def molecule(symbols, frame):
    return gto.M(atom=list(zip(symbols, frame["xyz"])), unit="Angstrom",
                 basis="def2-svp", charge=1, spin=1, verbose=0)


def run(args, model, symbols, frames, controls, profiled, sync, first_seed):
    """Create a new trajectory anchor on every pass, including each plain control."""
    # The CLI passes a null seed on its first call. An explicit same-geometry
    # anchor overrides only that call; subsequent guesses follow the CLI.
    def build(mol, family, **options):
        if first_seed is not None and not sequence.records:
            options["initial_reference"] = first_seed
        return workflow.build_reference(mol, family, density_fit=True, auxbasis=AUXBASIS,
                                        scf_backend=args.engine, scf_device=args.scf_device, **options)

    sequence = workflow._ReferenceSequence(
        "uhf", build=build, scf_args=controls, root_overlap_tolerance=0.85,
        seed_with_reference=args.engine == "generativeqc",
    )
    rows = []
    for frame in frames:
        t = Timers(sync)
        phase = "scf_and_root_tracking"
        try:
            with t.instrument() if profiled else nullcontext():
                with t.stage("other"):
                    mol = molecule(symbols, frame)
                    ref = sequence.build(mol)
                    phase = "energy_and_force"
                    result = workflow._evaluate_reference(ref, model, device=args.device, backend="zvector")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            return rows, {"line": frame["line"], "phase": phase, "error": str(error),
                          "attempt_s": t.wall, **(t.details() if profiled else {})}

        # Independent root diagnostics run after both the wall clock and the
        # hooks stop. This extra Fock must never be counted as a calculation call.
        fock = ref.get_hcore() + ref.get_veff(mol, ref.make_rdm1())
        residual = np.max(np.abs(fock @ ref.mo_coeff - (ref.get_ovlp() @ ref.mo_coeff)
                                 * ref.mo_energy[:, None, :]))
        s2 = float(ref.spin_square()[0])
        np.testing.assert_allclose(residual, 0, atol=1e-7, rtol=0)
        if abs(ref.e_tot - frame["HF_Eh"]) > 1e-8 or not np.isclose(s2, frame["S2"], atol=1e-5, rtol=0):
            return rows, {"line": frame["line"], "phase": "archive_root_check",
                          "error": "HF energy or spin differs from the archived branch",
                          "e_hf_Eh": float(ref.e_tot), "s2": s2, "canonical_residual": float(residual),
                          "attempt_s": t.wall, **(t.details() if profiled else {})}
        rows.append({
            "line": frame["line"], "wall_s": t.wall, **(t.details() if profiled else {}),
            "scf_cycles": ref.cycles if args.engine == "pyscf" else None,
            "guess": ("same_geometry_pyscf" if first_seed is not None and not rows
                      else sequence.records[-1]["initial_guess_source"]),
            "overlap": sequence.records[-1]["occupied_subspace_overlaps"],
            "canonical_residual": float(residual), "s2": s2,
            "e_hf_Eh": float(result["e_base"]), "e_total_Eh": float(result["e_tot"]),
            "force_Eh_per_Bohr": result["force"].tolist(),
        })
    return rows, None


def runtime_metadata(engine):
    """Record comparable runtime provenance for baseline and reuse measurements."""
    packages = ["pyscf", "torch", "numpy", "scipy"]
    if engine == "gpu4pyscf":
        packages.append("gpu4pyscf-cuda12x")
    elif engine == "generativeqc":
        packages.append("generativeqc")
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "no installed distribution metadata"
    return {"started_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(), "versions": versions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("pyscf", "gpu4pyscf", "generativeqc"))
    parser.add_argument("archive", type=Path, help="qm9_ie_geometry_opt directory")
    parser.add_argument("--scf-device", choices=("cpu", "cuda"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="spin network and orbital contractions (default: CPU)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="number of plain/profile pairs")
    parser.add_argument("--seed-initial", action="store_true",
                        help="anchor the first SCF with the checked PySCF root at the same geometry")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.threads) < 1:
        parser.error("threads and repeats must be positive")
    args.scf_device = args.scf_device or ("cpu" if args.engine == "pyscf" else "cuda")
    if Path(workflow.__file__).resolve().parents[2] != ROOT:
        raise RuntimeError("DeePHF must be imported from the checkout containing this script")
    lib.num_threads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    use_cuda = "cuda" in (args.scf_device, args.device)
    sync = torch.cuda.synchronize if use_cuda else lambda: None
    symbols, frames, digest = read_frames(args.archive)
    model = SpinCorrNet.load(CHECKPOINT).eval()
    controls = {"max_cycle": 300, "conv_tol": 1e-12}
    if args.engine == "generativeqc":
        controls["density_tolerance"] = 1e-10
    else:
        # The soft initial cation root stalls with the default DIIS history.
        controls["diis_space"] = 48
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {
        "commit": commit, "profile_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        **runtime_metadata(args.engine), "cpu": cpu, "engine": args.engine,
        "scf_device": args.scf_device, "response_device": args.device, "backend": "zvector",
        "cuda_device": torch.cuda.get_device_name() if use_cuda else None,
        "cuda_available": torch.cuda.is_available(),
        "importable_modules": {name: util.find_spec(name) is not None
                               for name in ("gpu4pyscf", "generativeqc")},
        "pyscf_threads": lib.num_threads(), "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_env": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
        "family": "DF-UHF", "basis": "def2-svp", "auxbasis": AUXBASIS, "scf_args": controls,
        "symbols": symbols, "unit": "Angstrom", "charge": 1, "spin": 1,
        "trajectory_sha256": digest, "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "frames": [{k: f.get(k) for k in ("line", "event", "stage", "step", "attempt", "HF_Eh", "S2")}
                   for f in frames], "runs": [],
    }
    first_seed = None
    if args.seed_initial:
        start = perf_counter()
        first_seed = workflow.build_reference(
            molecule(symbols, frames[0]), "uhf", scf_backend="pyscf", density_fit=True,
            auxbasis=AUXBASIS, scf_args={"conv_tol": 1e-12, "diis_space": 48, "max_cycle": 300},
        )
        report["initial_anchor"] = {"engine": "pyscf", "prepare_s": perf_counter() - start,
                                    "e_hf_Eh": float(first_seed.e_tot), "cycles": first_seed.cycles}
        np.testing.assert_allclose(first_seed.e_tot, frames[0]["HF_Eh"], atol=1e-8, rtol=0)
    max_difference = {"e_hf_Eh": 0.0, "e_total_Eh": 0.0, "force_Eh_per_Bohr": 0.0}
    baseline = None
    for repeat in range(args.repeats):
        for profiled in ((False, True) if repeat % 2 == 0 else (True, False)):
            rows, failure = run(args, model, symbols, frames, controls, profiled, sync, first_seed)
            if baseline is None:
                baseline = rows
            for ref, row in zip(baseline, rows):
                for key, tol in (("e_hf_Eh", 1e-9), ("e_total_Eh", 1e-9),
                                 ("force_Eh_per_Bohr", 1e-7)):
                    np.testing.assert_allclose(row[key], ref[key], atol=tol, rtol=0, equal_nan=False)
                    delta = np.max(np.abs(np.subtract(row[key], ref[key])))
                    max_difference[key] = max(max_difference[key], float(delta))
            report["runs"].append({"repeat": repeat + 1, "profiled": profiled,
                                   "frames": rows, "failed_frame": failure})
        # Keep the matched pair of attempts but do not time a different root or
        # silently substitute another parent for a failed continuation.
        if any(r["failed_frame"] for r in report["runs"]):
            break
    if args.engine == "generativeqc":
        from generativeqc._native import load_library

        library = load_library(device=args.scf_device)
        report["generativeqc_source_identity"] = library.generativeqc_get_source_identity().decode()
    common = min(len(r["frames"]) for r in report["runs"])
    report["compared_frame_count"] = common
    report["max_abs_difference_from_plain"] = max_difference if common else None
    report["status"] = "complete" if common == len(frames) else "incomplete"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] == "incomplete":
        raise SystemExit(f"Trajectory stopped early; see {args.output} for the failed frame.")


if __name__ == "__main__":
    main()
