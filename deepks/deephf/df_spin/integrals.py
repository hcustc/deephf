"""Density-fitted Coulomb/exchange integrals and their nuclear derivatives."""

from contextlib import contextmanager

import numpy as np
import torch
from pyscf import df, lib
from scipy.linalg import cho_factor, cho_solve


def auxiliary_molecule(reference):
    """Recover the auxiliary basis even when PySCF reads disk-backed factors."""
    backend = reference.with_df
    aux = backend.auxmol
    if aux is None:
        aux = df.addons.make_auxmol(reference.mol, backend.auxbasis)
    if backend.get_naoaux() != aux.nao:
        raise ValueError("spin gradients require a full-rank auxiliary metric")
    return aux


class DFFockDerivative:
    """Reuse three-centre and metric derivatives across orbital adjoints."""

    def __init__(self, reference):
        self.mol = reference.mol
        aux = auxiliary_molecule(reference)
        mol = self.mol
        self.basis_atoms = self._ao_atom_offsets(mol)
        self.aux_atoms = self._ao_atom_offsets(aux)
        self.ip1 = df.incore.aux_e2(mol, aux, intor="int3c2e_ip1", comp=3)
        self.metric_ip1 = aux.intor("int2c2e_ip1", comp=3)
        hcore_gradient = reference.nuc_grad_method().hcore_generator(mol)
        self.hcore1 = np.stack([hcore_gradient(a) for a in range(mol.natm)])
        metric = self._metric_solver(aux.intor("int2c2e", hermi=1))
        self.b = df.incore.aux_e2(mol, aux).reshape(-1, aux.nao)
        density = np.asarray(reference.make_rdm1()).reshape(-1, mol.nao, mol.nao)
        self.total = density.sum(0)
        self.total_coeff = metric(self.b.T @ self.total.ravel())
        self.metric = metric
        nspin = len(density)
        orbital = np.asarray(reference.mo_coeff).reshape(nspin, mol.nao, -1)
        occ = np.asarray(reference.mo_occ).reshape(nspin, -1)
        self.occupied = [c[:, o > 0] * np.sqrt(o[o > 0]) for c, o in zip(orbital, occ)]
        # Move the auxiliary index first for a compact occupied projection.
        three_center = self.b.T.reshape(aux.nao, mol.nao, mol.nao)
        self.fitted = [
            metric((three_center @ c).reshape(aux.nao, -1)).reshape(aux.nao, mol.nao, -1)
            for c in self.occupied
        ]
        self.exchange = 1.0 if nspin == 2 else 0.5

    @staticmethod
    def _ao_atom_offsets(mol):
        return mol.aoslice_by_atom()[:, 2]

    @staticmethod
    def _metric_solver(metric):
        """Factor the Coulomb metric used in the full-rank DF energy."""
        factor = cho_factor(metric, lower=True, check_finite=False)
        return lambda rhs: cho_solve(factor, rhs, check_finite=False)

    def contract(self, probe):
        """Return the derivative of Tr[probe F] at fixed AO probe and density."""
        total_probe = probe.sum(0)
        coeff_probe = self.metric(self.b.T @ total_probe.ravel())
        d_three = self.total[..., None] * coeff_probe + total_probe[..., None] * self.total_coeff
        d_metric = -np.outer(coeff_probe, self.total_coeff)
        for density, c, fit in zip(probe, self.occupied, self.fitted):
            zfit = density @ fit
            d_three -= (2 * self.exchange) * np.moveaxis(zfit @ c.T, 0, -1)
            d_metric += self.exchange * (zfit.reshape(len(fit), -1) @ fit.reshape(len(fit), -1).T)
        d_three = 0.5 * (d_three + d_three.transpose(1, 0, 2))
        d_metric = 0.5 * (d_metric + d_metric.T)
        # Three-centre integrals are translationally invariant. The auxiliary
        # motion is minus the sum of the two AO motions (equal by AO symmetry).
        ao_mov = np.einsum("xijp,ijp->xi", self.ip1, d_three)
        aux_mov = np.einsum("xijp,ijp->xp", self.ip1, d_three)
        aux_metric = np.einsum("xpq,pq->xp", self.metric_ip1, d_metric)
        ao_term = np.add.reduceat(ao_mov, self.basis_atoms, axis=1)
        aux_term = np.add.reduceat(aux_mov - aux_metric, self.aux_atoms, axis=1)
        return np.einsum("sij,axij->ax", probe, self.hcore1) + 2 * (aux_term - ao_term).T


@contextmanager
def device_df_jk(reference, device="cuda"):
    """Place the fixed-geometry DF factors on a device during SCF or response.

    The original PySCF backend is restored when the context exits. Other
    integral kernels and complex/non-Hermitian densities keep using PySCF.
    Changing geometry requires leaving the context and rebuilding the factors.
    """
    backend = getattr(reference, "with_df", None)
    if backend is None or getattr(reference, "only_dfj", False):
        raise ValueError("device_df_jk requires a reference with DF J and K")
    fallback = backend.get_jk
    factors = torch.cat(
        [torch.as_tensor(lib.unpack_tril(block), device=device) for block in backend.loop()]
    )
    source = backend._cderi
    nao = factors.shape[-1]
    flat = factors.flatten(1)

    @torch.no_grad()
    def get_jk(dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13, omega=None):
        if backend._cderi is not source:
            raise ValueError(
                "DF factors changed during device_df_jk; rebuild for the current geometry"
            )
        if omega is not None or hermi != 1 or np.asarray(dm).dtype != np.float64:
            return fallback(dm, hermi, with_j, with_k, direct_scf_tol, omega)
        density = torch.as_tensor(np.asarray(dm).reshape(-1, nao, nao), device=device)
        coulomb = exchange = None
        if with_j:
            coulomb = ((density.flatten(1) @ flat.T) @ flat).reshape(dm.shape).cpu().numpy()
        if with_k:
            # PySCF tags a density built from orbitals. Factoring through the
            # occupied orbitals reduces exchange cost during the SCF iteration.
            coeff, occ = getattr(dm, "mo_coeff", None), getattr(dm, "mo_occ", None)
            exchange = []
            if coeff is not None and occ is not None:
                for c, o in zip(
                    np.asarray(coeff).reshape(len(density), nao, -1),
                    np.asarray(occ).reshape(len(density), -1),
                ):
                    occupied = c[:, o > 0] * np.sqrt(o[o > 0])
                    projected = factors @ torch.as_tensor(occupied, device=device)
                    rows = projected.transpose(1, 2).reshape(-1, nao)
                    exchange.append(rows.T @ rows)
            else:
                rows = factors.reshape(-1, nao)
                for matrix in density:
                    exchange.append((matrix @ factors).reshape(-1, nao).T @ rows)
            exchange = torch.stack(exchange).reshape(dm.shape).cpu().numpy()
        return coulomb, exchange

    # Restore an existing instance override as well as the normal class method.
    had_override = "get_jk" in vars(backend)
    backend.get_jk = get_jk
    try:
        yield reference
    finally:
        if had_override:
            backend.get_jk = fallback
        else:
            del backend.get_jk
