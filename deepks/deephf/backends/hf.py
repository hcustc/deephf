"""External molecular HF solvers for DeePHF's real, collinear AO descriptors."""

import ctypes
from dataclasses import dataclass

import numpy as np
from pyscf import df, gto, scf
from scipy.linalg import eigh


@dataclass(frozen=True)
class HFState:
    """Accepted total energy (Hartree) and RHF or alpha/beta UHF AO density."""

    energy: float
    density: np.ndarray

    def validate(self, mol, restricted, overlap):
        """Check electron counts and occupied-projector normalization in the target AO metric."""
        expected_shape = (mol.nao, mol.nao) if restricted else (2, mol.nao, mol.nao)
        density = self.density
        if density.shape != expected_shape or np.iscomplexobj(density) or not (
            np.isfinite(self.energy) and np.isfinite(density).all()
        ):
            raise ValueError("invalid external HF energy or AO density")
        if not np.allclose(density, density.swapaxes(-1, -2), rtol=0, atol=1e-9):
            raise ValueError("external HF density is not symmetric")
        ds = density @ overlap
        counts = np.trace(ds, axis1=-2, axis2=-1)
        if not np.allclose(counts, mol.nelectron if restricted else mol.nelec, atol=1e-7, rtol=0):
            raise ValueError("external HF density has inconsistent AO normalization or spin counts")
        if not np.allclose(ds @ density, (2 if restricted else 1) * density, atol=1e-7, rtol=1e-8):
            raise ValueError("external HF density is not an occupied projector in the PySCF AO metric")

    def response_reference(self, reference):
        """Transport the occupied subspaces to the existing PySCF response code.

        Diagonalize each density in the AO metric, then canonicalize *within*
        each occupied/virtual space. A fresh Aufbau solve could select a
        different open-shell root. No PySCF SCF iterations are performed.
        """
        overlap = reference.get_ovlp()
        if hasattr(reference, "xc") or not isinstance(reference, (scf.hf.RHF, scf.uhf.UHF)):
            raise TypeError("orbital transport requires a molecular RHF or UHF reference")
        restricted = not isinstance(reference, scf.uhf.UHF)
        self.validate(reference.mol, restricted, overlap)
        blocks = self.density[None] if restricted else self.density
        weight = 2 if restricted else 1
        orbitals, occupations = [], []
        for density in blocks:
            values, vectors = eigh(overlap @ density @ overlap, overlap)
            occupied = values[::-1] > weight / 2
            orbitals.append(vectors[:, ::-1])
            occupations.append(weight * occupied.astype(float))
        coeff = orbitals[0] if restricted else np.stack(orbitals)
        occ = occupations[0] if restricted else np.stack(occupations)
        hcore = reference.get_hcore()
        veff = reference.get_veff(reference.mol, self.density)
        energy = reference.energy_tot(self.density, hcore, veff)
        if abs(energy - self.energy) > 1e-8 or not np.isfinite(energy):
            raise ValueError("external HF energy is inconsistent with the PySCF response Hamiltonian")
        fock = hcore + veff
        eps, coeff = reference.canonicalize(coeff, occ, fock)
        if np.max(np.abs(fock @ coeff - (overlap @ coeff) * eps[..., None, :])) > 1e-7:
            raise ValueError("external HF density is not stationary for the PySCF response Hamiltonian")
        if not np.allclose(reference.make_rdm1(coeff, occ), self.density, atol=1e-7, rtol=0):
            raise ValueError("canonicalization changed the external HF density")
        reference.mo_occ, reference.mo_coeff, reference.mo_energy = occ, coeff, eps
        reference.e_tot, reference.converged = self.energy, True
        return reference


def _shells(molecule):
    """Export actual contractions in PySCF AO order, including per-atom overrides."""
    from generativeqc import Primitive, Shell

    shells = []
    for i in range(molecule.nbas):
        exponents = molecule.bas_exp(i)
        for coeff in molecule.bas_ctr_coeff(i).T:
            primitives = tuple(Primitive(e, c) for e, c in zip(exponents, coeff) if c != 0)
            shells.append(Shell(molecule.bas_atom(i), molecule.bas_angular(i), primitives))
    return tuple(shells)


def validate_seed(seed, target, family):
    """A warm start may move nuclei but must describe the same ordered HF problem."""
    if (not isinstance(seed, (scf.hf.RHF, scf.uhf.UHF)) or hasattr(seed, "xc")
            or not seed.converged or np.iscomplexobj(seed.mo_coeff)):
        raise ValueError("a warm start requires a converged molecular HF reference")
    source, mol = seed.mol, target.mol
    if (family not in ("rhf", "uhf") or (seed.mo_occ.ndim == 1) != (family == "rhf")
            or source.charge != mol.charge or source.spin != mol.spin or source.cart != mol.cart
            or source.omega != mol.omega or source.has_ecp() or mol.has_ecp()
            or source.pseudo or mol.pseudo or source.nucmod or mol.nucmod
            or source.elements != mol.elements or source.ao_labels() != mol.ao_labels()
            or not gto.same_basis_set(source, mol) or not np.isfinite(seed.e_tot)):
        raise ValueError("warm-start charge, spin, ordered nuclei or orbital basis mismatch")
    src_df, dst_df = getattr(seed, "with_df", None), getattr(target, "with_df", None)
    if (src_df is None) != (dst_df is None) or getattr(seed, "only_dfj", False):
        raise ValueError("warm-start density-fitting Hamiltonian mismatch")
    if dst_df is not None and not gto.same_basis_set(
        df.addons.make_auxmol(source, src_df.auxbasis), df.addons.make_auxmol(mol, dst_df.auxbasis)
    ):
        raise ValueError("warm-start auxiliary basis mismatch")


def projected_seed(seed, mol):
    """Project occupied AOs for a PySCF-family SCF guess at the new geometry."""
    if seed is None:
        return None
    density = np.asarray(seed.make_rdm1())
    if np.array_equal(seed.mol.atom_coords(), mol.atom_coords()):
        return density
    return scf.addons.project_dm_nr2nr(seed.mol, density, mol)


def _seeded_generativeqc(calc, mol, seed):
    """Use the public HF warm-state C ABI on a single Python prepared batch.

    Supplies the source geometry to the native metric checks. Only the density
    returned after a successful target SCF is used for DeePHF.
    """
    from generativeqc import _native

    xyz = np.ascontiguousarray(seed.mol.atom_coords())
    target_xyz = mol.atom_coords()
    density = np.ascontiguousarray(seed.make_rdm1(), dtype=np.float64)
    if mol.spin < 0:
        density = np.ascontiguousarray(density[::-1])
    pointer = ctypes.POINTER(ctypes.c_double)

    def descriptor():
        return _native.HfWarmState(struct_size=ctypes.sizeof(_native.HfWarmState),
                                   abi_version=_native.ABI_VERSION)

    atoms = list(zip(mol.atom_charges().tolist(), xyz))
    with calc.prepare_batch([atoms], charges=[mol.charge], multiplicities=[abs(mol.spin) + 1],
                            warm_start=True) as batch:
        # GenerativeQC exposes source geometry and spin-resolved restart
        # buffers through the native handle of its Python prepared owner.
        api, handle = batch._library, batch._batch
        incoming = descriptor()
        incoming.density, incoming.density_count = density.ctypes.data_as(pointer), density.size
        incoming.coordinates, incoming.coordinate_count = xyz.ctypes.data_as(pointer), xyz.size
        incoming.present, incoming.energy = 1, float(seed.e_tot)
        _native.check(api, api.generativeqc_batch_restore_hf_warm_states(
            handle, ctypes.byref(incoming), 1))
        result = batch.execute([target_xyz], properties=("energy",), strict=True).items[0]
        if not result.converged or not result.warm_start_used or result.warm_start_fallback:
            raise RuntimeError("GenerativeQC did not converge using the requested HF seed")
        outgoing = descriptor()
        get = api.generativeqc_batch_get_hf_warm_state
        _native.check(api, get(handle, 0, ctypes.byref(outgoing)))
        if not outgoing.present or outgoing.density_count != density.size or outgoing.coordinate_count != xyz.size:
            raise RuntimeError("GenerativeQC returned incompatible warm-state buffers")
        final, geometry = np.empty_like(density), np.empty_like(xyz)
        outgoing.density, outgoing.coordinates = final.ctypes.data_as(pointer), geometry.ctypes.data_as(pointer)
        _native.check(api, get(handle, 0, ctypes.byref(outgoing)))
        if not np.allclose(geometry, target_xyz, rtol=0, atol=1e-12):
            raise RuntimeError("GenerativeQC returned a density at a different geometry")
    return HFState(result.energy, final[::-1].copy() if mol.spin < 0 else final)


def _generativeqc(reference, restricted, device, controls, seed):
    from generativeqc import Calculator

    mol = reference.mol
    if mol.cart or mol.has_ecp() or mol.pseudo or np.any(mol.atom_charges() == 0):
        raise ValueError("GenerativeQC density export requires all-electron spherical AOs on real atoms")
    if mol.symmetry or mol.omega or np.any(mol._atm[:, gto.NUC_MOD_OF] != gto.NUC_POINT):
        raise ValueError("GenerativeQC requires point nuclei, a full Coulomb potential and unconstrained symmetry")
    unknown = controls.keys() - {"conv_tol", "max_cycle", "density_tolerance"}
    if unknown:
        raise ValueError(f"unsupported GenerativeQC SCF controls: {sorted(unknown)}")
    kwargs = {}
    if getattr(reference, "with_df", None) is not None:
        # The engines have different default fitting bases. Use the basis
        # selected by PySCF for this exact molecule, rather than just its name.
        aux = df.addons.make_auxmol(mol, reference.with_df.auxbasis)
        kwargs.update(density_fitting=device, auxiliary_basis=_shells(aux))
    calc = Calculator(
        method="rhf" if restricted else "uhf", device=device, basis=_shells(mol),
        basis_representation="spherical", precision="fp64", screening_tolerance=1e-14,
        energy_tolerance=controls.get("conv_tol", 1e-12),
        density_tolerance=controls.get("density_tolerance", 1e-10),
        max_iterations=controls.get("max_cycle", 100), **kwargs,
    )
    if seed is not None:
        return _seeded_generativeqc(calc, mol, seed)
    result = calc.singlepoint(
        list(zip(mol.atom_charges().tolist(), mol.atom_coords(unit="Bohr"))),
        charge=mol.charge, multiplicity=abs(mol.spin) + 1, properties=("energy", "density"),
    )
    if not result.converged:
        raise RuntimeError("GenerativeQC SCF did not converge")
    total, spin = result.density.matrix, result.density.spin_matrix
    if restricted:
        if spin is not None:
            raise ValueError("GenerativeQC returned an unrestricted density for RHF")
        density = total.copy()
    else:
        if spin is None:
            raise ValueError("GenerativeQC did not return the UHF spin density")
        if mol.spin < 0:
            spin = -spin
        density = np.stack(((total + spin) / 2, (total - spin) / 2))
    return HFState(result.energy, np.asarray(density))


def solve_external(reference, family, engine, device, controls, seed):
    """Run one external SCF and check its AO/spin conventions before projection."""
    if family not in ("rhf", "uhf"):
        raise ValueError("external DeePHF SCF engines support RHF and UHF")
    restricted = family == "rhf"
    if engine == "gpu4pyscf":
        if device != "cuda":
            raise ValueError("GPU4PySCF requires scf_device=cuda")
        import gpu4pyscf  # noqa: F401 -- registers PySCF's GPU conversion

        gpu = reference.to_gpu()
        if getattr(gpu, "with_df", None) is not None and gpu.with_df.auxbasis is None:
            gpu.with_df.auxbasis = df.addons.make_auxbasis(reference.mol)
        # GPU4PySCF tests the norm over all orbital gradients. On DF-UHF
        # this can plateau at a few 1e-8 even for a stationary FP64 root.
        gpu.conv_tol_grad = controls.get("conv_tol_grad", 5e-8)
        gpu.kernel(dm0=projected_seed(seed, reference.mol))
        if not gpu.converged:
            raise RuntimeError("GPU4PySCF SCF did not converge")
        state = HFState(float(gpu.e_tot), gpu.make_rdm1().get())
    elif engine == "generativeqc":
        if device not in ("cpu", "cuda"):
            raise ValueError("GenerativeQC scf_device must be cpu or cuda")
        state = _generativeqc(reference, restricted, device, controls, seed)
    else:
        raise ValueError(f"unknown SCF backend {engine!r}")
    return state
