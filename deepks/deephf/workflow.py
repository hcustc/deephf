"""Public construction and inference workflow for strict DeePHF methods."""

import os

import numpy as np
import torch
from pyscf import dft, scf

from deepks.data.io import build_molecule, dump_data, iter_system
from deepks.model.model import CorrNet
from deepks.model.spin import SpinCorrNet
from deepks.utils import get_sys_name, load_basis, load_sys_paths

from .backends.hf import HFState, projected_seed, solve_external, validate_seed
from .method import DeePHF
from .rks_method import RKSDeePHF
from .df_spin import DFSpinDeePHF
from .unrestricted_method import UHFDeePHF, UKSDeePHF


METHOD_CLASSES = {
    scf.hf.RHF: DeePHF,
    scf.uhf.UHF: UHFDeePHF,
    dft.rks.RKS: RKSDeePHF,
    dft.uks.UKS: UKSDeePHF,
}
REFERENCE_FAMILIES = frozenset(("rhf", "uhf", "rks", "uks"))


def make_deephf(reference, model, *, projector_basis=None, device="cpu", response_options=None, adjoint_options=None):
    """Construct the exact public DeePHF method matching a native reference."""
    if isinstance(model, SpinCorrNet):
        if response_options or adjoint_options:
            raise ValueError("custom response options are unsupported for spin gradients")
        if projector_basis is not None and load_basis(projector_basis) != model.network._pbas:
            raise ValueError("projector basis differs from the SpinCorrNet checkpoint")
        return DFSpinDeePHF(reference, model, device=device)
    method_class = METHOD_CLASSES.get(type(reference))
    if method_class is None:
        raise TypeError("the reference type has no strict DeePHF method")
    if projector_basis is None and model is not None:
        projector_basis = model._pbas
    return method_class(
        reference,
        model,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )


def _new_reference(molecule, family, scf_args, verbose, density_fit, auxbasis):
    """Prepare the common molecular Hamiltonian without starting SCF."""
    if auxbasis is not None and not density_fit:
        raise ValueError("auxbasis requires density_fit=True")
    if type(family) is not str or family.lower() not in REFERENCE_FAMILIES:
        raise ValueError("reference family must be one of rhf, uhf, rks, or uks")
    family = family.lower()
    if family in {"rhf", "rks"} and molecule.spin != 0:
        raise ValueError(f"the strict {family.upper()} family requires molecular spin zero")
    if family == "rhf":
        reference = scf.RHF(molecule)
    elif family == "uhf":
        reference = scf.UHF(molecule)
    elif family == "rks":
        reference = dft.RKS(molecule)
    else:
        reference = dft.UKS(molecule)
    expected_type = {"rhf": scf.hf.RHF, "uhf": scf.uhf.UHF, "rks": dft.rks.RKS, "uks": dft.uks.UKS}[family]
    if type(reference) is not expected_type:
        raise TypeError(f"the requested {family.upper()} constructor did not produce its exact native reference type")
    if density_fit:
        if family not in {"rhf", "uhf"}:
            raise ValueError("the spin DF-HF workflow requires RHF or UHF")
        reference = reference.density_fit(auxbasis=auxbasis)
    reference.verbose = verbose
    controls = {
        "conv_tol": 1.0e-13,
        "conv_tol_grad": 1.0e-10,
        "conv_tol_cpscf": 1.0e-12,
        "max_cycle": 100,
    }
    if density_fit:
        # Tight total-energy stopping alone can stall on a converged UHF
        # branch. Check canonical physical-Fock stationarity at the force API.
        controls.update(conv_tol=1e-12, conv_tol_grad=1e-8)
    controls.update({} if scf_args is None else dict(scf_args))
    supported_controls = {
        "conv_tol",
        "conv_tol_grad",
        "conv_tol_cpscf",
        "max_cycle",
        "diis_space",
        "level_shift",
        "direct_scf",
        "conv_check",
    }
    unknown = sorted(set(controls) - supported_controls)
    if unknown:
        raise ValueError("unsupported strict reference controls: " + ", ".join(unknown))
    reference.set(**controls)
    if family in {"rks", "uks"}:
        reference.xc = "LDA_X + LDA_C_VWN"
        reference.grids.atom_grid = {symbol: (20, 50) for symbol in set(molecule.elements)}
        reference.grids.prune = None
        reference.grids.alignment = 1
        reference.grids.cutoff = 1.0e-15
        reference.grids.build(with_non0tab=True, sort_grids=False)
        reference.small_rho_cutoff = 0.0
    return reference


def _solve(molecule, family, scf_backend, scf_device, scf_args, verbose, density_fit, auxbasis,
           *, density_only=False, initial_reference=None):
    controls = dict(scf_args or {})
    native_controls = {} if scf_backend == "generativeqc" else controls
    reference = _new_reference(molecule, family, native_controls, verbose, density_fit, auxbasis)
    if initial_reference is not None:
        validate_seed(initial_reference, reference, family.lower())
    if scf_backend == "pyscf":
        if scf_device not in (None, "cpu"):
            raise ValueError("PySCF requires scf_device=cpu")
        reference.kernel(dm0=projected_seed(initial_reference, molecule))
        if not reference.converged:
            raise RuntimeError(f"the native {family.upper()} reference did not converge")
        if density_only:
            return HFState(reference.e_tot, np.asarray(reference.make_rdm1()))
        return reference
    state = solve_external(reference, family.lower(), scf_backend,
                           "cuda" if scf_device is None else scf_device, controls, initial_reference)
    if density_only:
        state.validate(molecule, family.lower() == "rhf", reference.get_ovlp())
        return state
    return state.response_reference(reference)


def build_hf_state(molecule, family, *, scf_backend="pyscf", scf_device=None, density_fit=False,
                   auxbasis=None, scf_args=None, initial_reference=None, verbose=0):
    """Return HF total energy and AO density in the supplied molecule's basis.

    Density is spin-summed for RHF and ordered (alpha, beta) for UHF.
    The optional converged seed may come from another supported HF engine.
    """
    if not isinstance(family, str) or family.lower() not in ("rhf", "uhf"):
        raise ValueError("an HF density requires the RHF or UHF family")
    return _solve(molecule, family, scf_backend, scf_device, scf_args, verbose,
                  density_fit, auxbasis, density_only=True, initial_reference=initial_reference)


def build_reference(molecule, family, *, scf_args=None, verbose=0, density_fit=False,
                    auxbasis=None, scf_backend="pyscf", scf_device=None, initial_reference=None):
    """Return canonical PySCF response orbitals for the selected SCF engine's root."""
    return _solve(molecule, family, scf_backend, scf_device, scf_args, verbose, density_fit, auxbasis,
                  initial_reference=initial_reference)


def _state_energies(base, correction):
    return {"converged": np.asarray(True), "e_base": np.asarray(base),
            "e_corr": np.asarray(correction), "e_tot": np.asarray(base + correction)}


def evaluate_molecule(
    molecule,
    model,
    *,
    family,
    backend="direct",
    projector_basis=None,
    device="cpu",
    scf_args=None,
    response_options=None,
    adjoint_options=None,
    verbose=0,
    energy_only=False,
    density_fit=False,
    auxbasis=None,
    scf_backend="pyscf",
    scf_device=None,
    initial_reference=None,
):
    """Evaluate one state, optionally omitting derivative and descriptor outputs."""
    spin_model = isinstance(model, SpinCorrNet)
    bare_energy = model is None and energy_only
    if density_fit and not (spin_model or bare_energy):
        raise ValueError("density_fit requires a SpinCorrNet checkpoint or a bare HF energy")
    if spin_model and projector_basis is not None and load_basis(projector_basis) != model.network._pbas:
        raise ValueError("projector basis differs from the SpinCorrNet checkpoint")
    density_model = spin_model or (bare_energy and (density_fit or scf_backend != "pyscf"))
    if energy_only and density_model and isinstance(family, str) and family.lower() in ("rhf", "uhf"):
        state = build_hf_state(molecule, family, scf_backend=scf_backend, scf_device=scf_device,
                               scf_args=scf_args, verbose=verbose, density_fit=density_fit,
                               auxbasis=auxbasis, initial_reference=initial_reference)
        corr = 0.0
        if spin_model:
            corr = model.to(device).eval().correction_from_density(molecule, state.density)
        return _state_energies(state.energy, corr)
    reference = build_reference(molecule, family, scf_backend=scf_backend, scf_device=scf_device,
                                scf_args=scf_args, verbose=verbose, density_fit=density_fit,
                                auxbasis=auxbasis, initial_reference=initial_reference)
    if spin_model:
        correction = model.to(device).eval().correction_energy(reference)
        result = _state_energies(reference.e_tot, correction)
        if not energy_only:
            method = make_deephf(reference, model, projector_basis=projector_basis, device=device,
                                 response_options=response_options, adjoint_options=adjoint_options)
            gradient = method.gradient(backend=backend)
            result.update(descriptor=method.descriptor(), gradient=gradient, force=-gradient)
        return result
    method = make_deephf(
        reference,
        model,
        projector_basis=projector_basis,
        device=device,
        response_options=response_options,
        adjoint_options=adjoint_options,
    )
    with method._controlled_calculation():
        energy = float(method.kernel())
        if energy_only:
            return {"converged": np.asarray(True), "e_base": np.asarray(method.e_base),
                    "e_corr": np.asarray(method.e_corr), "e_tot": np.asarray(energy)}
        gradient_driver = method.nuc_grad_method(
            backend=backend,
            retain_details=False,
        )
        gradient = np.asarray(gradient_driver.kernel())
        descriptor = np.asarray(method.descriptor(), dtype=np.float64)
        result = {
            "converged": np.asarray(True),
            "e_base": np.asarray(method.e_base, dtype=np.float64),
            "e_corr": np.asarray(method.e_corr, dtype=np.float64),
            "e_tot": np.asarray(energy, dtype=np.float64),
            "descriptor": descriptor,
            "gradient": gradient,
            "force": -gradient,
        }
    return result


def main(
    systems,
    *,
    reference,
    model_file=None,
    basis="sto-3g",
    projector_basis=None,
    backend="direct",
    device="cpu",
    dump_dir=".",
    mol_args=None,
    scf_args=None,
    response_options=None,
    adjoint_options=None,
    verbose=0,
    energy_only=False,
    density_fit=False,
    auxbasis=None,
    scf_backend="pyscf",
    scf_device=None,
):
    """Run strict DeePHF inference for every frame and persist canonical outputs."""
    if model_file is None or str(model_file).upper() == "NONE":
        model = None
    else:
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
        if checkpoint.get("format") in SpinCorrNet.CHECKPOINT_FORMATS:
            model = SpinCorrNet.load_dict(checkpoint)
        else:
            model = CorrNet.load_dict(checkpoint, strict=True).double().eval()
    mol_args = {} if mol_args is None else dict(mol_args)
    outputs = []
    for system_path in load_sys_paths(systems):
        frames = []
        for atoms, attributes, _labels in iter_system(system_path):
            molecule_input = {
                **mol_args,
                "atom": atoms,
                "basis": basis,
                "verbose": verbose,
                **attributes,
            }
            molecule = build_molecule(**molecule_input)
            frames.append(
                evaluate_molecule(
                    molecule,
                    model,
                    family=reference,
                    backend=backend,
                    projector_basis=projector_basis,
                    device=device,
                    scf_args=scf_args,
                    response_options=response_options,
                    adjoint_options=adjoint_options,
                    verbose=verbose,
                    energy_only=energy_only,
                    density_fit=density_fit,
                    auxbasis=auxbasis,
                    scf_backend=scf_backend,
                    scf_device=scf_device,
                )
            )
        if not frames:
            raise ValueError(f"system {system_path!r} contains no molecular frames")
        collected = {name: np.asarray([frame[name] for frame in frames]) for name in frames[0]}
        output_directory = os.path.join(dump_dir, get_sys_name(system_path))
        if energy_only:
            # Keep earlier gradient files from being associated with new energies.
            output_directory = os.path.join(output_directory, "energy_only")
        dump_data(output_directory, **collected)
        outputs.append((output_directory, collected))
    return outputs


__all__ = ["build_hf_state", "build_reference", "evaluate_molecule", "main", "make_deephf"]
