"""Occupied–virtual linear response of canonical, density-fitted RHF/UHF."""

import numpy as np
import torch
from pyscf import lib
from scipy.sparse.linalg import LinearOperator, gmres


def _check_direct_residual(error, rhs):
    """Bound each Cartesian residual independently in the infinity norm."""
    scale = rhs.abs().amax(dim=1).clamp_min(1.0)
    residual = (error.abs().amax(dim=1) / scale).max().item()
    if not np.isfinite(residual) or residual > 1e-8:
        raise RuntimeError(f"DF-HF orbital response scaled residual {residual:.3g}")


class DFOrbitalResponse:
    """Use the same metric-whitened DF factors for direct and adjoint solves.

    Each spin block uses virtual-major, occupied-minor order. RHF uses one
    doubly occupied block. UHF uses two singly occupied blocks even when
    the nominal molecular spin is zero.
    """

    def __init__(self, reference, device, *, prepare_factors=True):
        self.reference = reference
        self.device = torch.device(device)
        self.restricted = np.ndim(reference.mo_occ) == 1
        self.weight = 2 if self.restricted else 1
        coeff = np.asarray(reference.mo_coeff)
        n = reference.mol.nao
        occ = np.atleast_2d(reference.mo_occ)
        energy = np.atleast_2d(reference.mo_energy)
        self.occupied, self.virtual, self.energies, self.gaps = [], [], [], []
        for c, mask, e in zip(coeff.reshape(len(occ), n, -1), occ > 0, energy):
            self.occupied.append(self.tensor(c[:, mask]))
            self.virtual.append(self.tensor(c[:, ~mask]))
            self.energies.append(self.tensor(e[mask]))
            self.gaps.append(self.tensor(e[~mask, None] - e[mask]))

        self.sizes = [gap.numel() for gap in self.gaps]
        self.diagonal = torch.cat([g.flatten() for g in self.gaps])
        if not prepare_factors:
            # External Fock operators supply both VO and OO responses directly.
            return

        # Only MO blocks are retained after transforming the fitted AO factors.
        packed = reference.with_df.loop()
        parts = [[] for _ in self.gaps]
        for block in packed:
            factors = self.tensor(lib.unpack_tril(block))
            for matrices, o, v in zip(parts, self.occupied, self.virtual):
                matrices.append((o.T @ factors @ o, v.T @ factors @ o, v.T @ factors @ v))
        self.blocks = [tuple(torch.cat(columns) for columns in zip(*rows)) for rows in parts]
        self.vo = torch.cat([vo.flatten(1) for _, vo, _ in self.blocks], dim=1)

    def tensor(self, value):
        return torch.as_tensor(np.ascontiguousarray(value), device=self.device)

    def action(self, flat):
        """Apply the symmetric Hessian without constructing its dense matrix."""
        coulomb = (2 * self.weight) * (self.vo.T @ (self.vo @ flat))
        out = []
        for x, gap, (oo, vo, vv) in zip(flat.split(self.sizes), self.gaps, self.blocks):
            x = x.reshape_as(gap)
            # Exchange term (ab|ij) X_bj + (aj|bi) X_bj.
            exchange = ((vv @ x) @ oo + (vo @ x.T) @ vo).sum(0)
            out.append((gap * x - exchange).flatten())
        return torch.cat(out) + coulomb

    def adjoint(self, rhs):
        """Solve the scalar adjoint with preconditioned GMRES and check residual."""
        rhs = torch.cat([r.flatten() for r in rhs])
        norm = torch.linalg.vector_norm(rhs).item()
        if norm == 0:
            return torch.zeros_like(rhs)
        diagonal = self.diagonal.abs().clamp_min(1e-3).cpu().numpy()
        shape = (len(rhs), len(rhs))
        operator = LinearOperator(
            shape,
            matvec=lambda x: self.action(self.tensor(x)).cpu().numpy(),
            dtype=np.float64,
        )
        preconditioner = LinearOperator(shape, matvec=lambda x: x / diagonal, dtype=np.float64)
        dual, status = gmres(
            operator,
            rhs.cpu().numpy() / norm,
            M=preconditioner,
            rtol=1e-9,
            atol=0,
            restart=32,
            maxiter=8,
        )
        dual = self.tensor(dual) * norm
        if status or torch.linalg.vector_norm(self.action(dual) - rhs).item() > 2e-9 * norm:
            raise RuntimeError(f"DF-HF orbital adjoint did not converge (GMRES status {status})")
        return dual

    def occupied_potential(self, dual):
        """Occupied blocks of the Fock response to symmetrized VO amplitudes."""
        coulomb = self.vo @ dual
        result = []
        for (oo, vo, _), gap, x in zip(self.blocks, self.gaps, dual.split(self.sizes)):
            exchange = (oo @ (x.reshape_as(gap).T @ vo)).sum(0)
            result.append(
                self.weight * (coulomb @ oo.flatten(1)).reshape(oo.shape[1:])
                - 0.5 * (exchange + exchange.T)
            )
        return result

    def overlap_motion(self):
        """Derivative of the AO metric, with the nuclear coordinates flattened."""
        mol = self.reference.mol
        deriv = -mol.intor("int1e_ipovlp", comp=3)
        half = np.zeros((mol.natm, *deriv.shape))
        for atom, (_, _, begin, end) in enumerate(mol.aoslice_by_atom()):
            half[atom, :, begin:end] = deriv[:, begin:end]
        return self.tensor((half + half.swapaxes(-1, -2)).reshape(-1, mol.nao, mol.nao))

    def density_derivatives(self):
        """Solve all nuclear perturbations together for the relaxed AO densities."""
        ref = self.reference
        s1 = self.overlap_motion()
        driver = ref.Hessian()
        # PySCF differentiates the AO and auxiliary centres and the Coulomb
        # metric when forming the explicit first-order DF Fock matrices.
        driver.auxbasis_response = 1
        fock1 = driver.make_h1(ref.mo_coeff, ref.mo_occ)
        if self.restricted:
            fock1 = (fock1,)
        gauge = [-self.weight * o @ (o.T @ s1 @ o) @ o.T for o in self.occupied]
        occupied_metric = [o.T @ s1 @ o for o in self.occupied]
        # Coulomb and exchange responses to the occupied metric gauge.
        aux_charge = -self.weight * sum(
            metric.flatten(1) @ oo.flatten(1).T
            for metric, (oo, _, _) in zip(occupied_metric, self.blocks)
        )
        rhs, diagonal_blocks = [], []
        for o, v, eps, gap, (oo, vo, vv), f1, metric in zip(
            self.occupied,
            self.virtual,
            self.energies,
            self.gaps,
            self.blocks,
            fock1,
            occupied_metric,
        ):
            explicit = v.T @ self.tensor(np.asarray(f1).reshape(s1.shape)) @ o
            exchange = torch.einsum("Paj,xjk,Pki->xai", vo, metric, oo)
            motion = (v.T @ s1 @ o) * eps - explicit - exchange
            rhs.append(motion.flatten(1) - aux_charge @ vo.flatten(1))
            same_spin = torch.einsum("Pab,Pij->aibj", vv, oo) + torch.einsum(
                "Paj,Pbi->aibj", vo, vo
            )
            diagonal_blocks.append(
                torch.diag(gap.flatten()) - same_spin.reshape(gap.numel(), gap.numel())
            )
        rhs = torch.cat(rhs, dim=1)
        if rhs.shape[1] == 0:
            return [p.cpu() for p in gauge]
        hessian = torch.block_diag(*diagonal_blocks) + (2 * self.weight) * (self.vo.T @ self.vo)
        amplitudes = torch.linalg.solve(hessian, rhs.T).T
        _check_direct_residual(amplitudes @ hessian.T - rhs, rhs)
        result = []
        for p, v, o, gap, x in zip(
            gauge, self.virtual, self.occupied, self.gaps, amplitudes.split(self.sizes, dim=1)
        ):
            change = self.weight * v @ x.reshape(len(s1), *gap.shape) @ o.T
            result.append((p + change + change.transpose(-1, -2)).cpu())
        return result
