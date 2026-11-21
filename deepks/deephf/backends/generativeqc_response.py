"""Apply the GenerativeQC density-fitted J/K operator to a DeePHF adjoint."""

from contextlib import ExitStack, contextmanager

import numpy as np
import torch
from pyscf import gto

from .hf import _shells
from ..df_spin.integrals import auxiliary_molecule
from ..df_spin.orbitals import DFOrbitalResponse


class _FockResponse(DFOrbitalResponse):
    """Replace contracted MO factors with native J/K on each trial AO density."""

    def __init__(self, reference, device, plan):
        self.plan = plan
        super().__init__(reference, device, prepare_factors=False)

    def _induced(self, flat):
        # X is a virtual-by-occupied rotation. The symmetrized AO perturbation
        # has occupation weight two for RHF, one for each UHF spin block.
        changes = []
        for x, gap, v, o in zip(flat.split(self.sizes), self.gaps, self.virtual, self.occupied):
            vo = v @ x.reshape_as(gap) @ o.T
            changes.append((self.weight * (vo + vo.T)).cpu().numpy())
        result = self.plan.evaluate(changes[0] if self.restricted else np.stack(changes))
        exchange = np.asarray(result.exchange).reshape(len(changes), *changes[0].shape)
        return [self.tensor(result.coulomb - k / self.weight) for k in exchange]

    def action(self, flat):
        fields = self._induced(flat)
        vo = [v.T @ p @ o for v, p, o in zip(self.virtual, fields, self.occupied)]
        return self.diagonal * flat + torch.cat([block.flatten() for block in vo])

    def occupied_potential(self, dual):
        # The overlap gauge contracts one half of the symmetrized AO response.
        return [0.5 * o.T @ p @ o for o, p in zip(self.occupied, self._induced(dual))]


@contextmanager
def generativeqc_response(reference, device):
    """Own one native Fock plan throughout GMRES, with PySCF nuclear integrals."""
    device = torch.device(device)
    mol = reference.mol
    aux = auxiliary_molecule(reference)
    if mol.cart or mol.has_ecp() or mol.pseudo or mol.omega or np.any(mol.atom_charges() == 0):
        raise ValueError("native GenerativeQC response requires spherical all-electron molecular bases")
    if np.any(mol._atm[:, gto.NUC_MOD_OF] != gto.NUC_POINT):
        raise ValueError("native GenerativeQC response requires point nuclei")
    orbital_l = max(mol.bas_angular(i) for i in range(mol.nbas))
    auxiliary_l = max(aux.bas_angular(i) for i in range(aux.nbas))
    if max(orbital_l, auxiliary_l) > 3:
        raise ValueError(
            f"GenerativeQC FockPlan supports shells through f (l=3), but this calculation "
            f"has orbital l_max={orbital_l} and auxiliary l_max={auxiliary_l}. "
            "Use backend='zvector' (--backend zvector) with the same SCF engine and bases."
        )
    from generativeqc.fock import FockBuildSpec, FockPlan
    from generativeqc_compiler.dft import NativeAO

    atoms = list(zip(mol.atom_charges().tolist(), mol.atom_coords()))
    kwargs = dict(representation="spherical", charge=mol.charge, multiplicity=abs(mol.spin) + 1)
    spec = FockBuildSpec.hf(
        "restricted" if reference.mo_occ.ndim == 1 else "unrestricted",
        coulomb="density_fitted", exchange="density_fitted", derivative_order=0,
    )
    with ExitStack() as stack:
        orbital = stack.enter_context(NativeAO(atoms, basis=_shells(mol), **kwargs))
        fitting = stack.enter_context(NativeAO(atoms, basis=_shells(aux), **kwargs))
        ordinal = torch.cuda.current_device() if device.type == "cuda" and device.index is None else device.index
        plan = stack.enter_context(FockPlan(
            orbital, spec, auxiliary=fitting, device=device.type,
            device_id=ordinal or 0, screening_tolerance=0.0,
        ))
        if plan.diagnostics["auxiliary_rank"] != aux.nao:
            raise ValueError("GenerativeQC response and PySCF require the same full-rank fitting metric")
        yield _FockResponse(reference, device, plan)
