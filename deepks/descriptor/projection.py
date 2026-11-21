"""PySCF projection integrals and the shared atomic descriptor interface."""

from copy import deepcopy

import numpy as np
import torch
from pyscf import gto

from deepks.gpu import DEFAULT_CUDA_DEVICE, torch_from_array
from deepks.utils import get_shell_sec, load_basis

from .core import descriptor, dq_dP, projected_density
from .derivatives import (
    dD_dR_explicit,
    dq_dR_explicit,
    dq_dR_explicit_component,
    contract_descriptor_derivatives,
)


def is_ghost_atom(mol, atom_index: int) -> bool:
    """Return whether a PySCF atom has zero nuclear charge."""
    return int(mol.atom_charge(atom_index)) == 0


def descriptor_atom_indices(mol) -> tuple[int, ...]:
    """Return raw atom indices that carry descriptor projectors."""
    return tuple(
        atom_index
        for atom_index in range(mol.natm)
        if not is_ghost_atom(mol, atom_index)
    )


def spin_summed_ao_density(ao_density):
    """Return the AO density used by the spin-summed descriptor contract."""
    if not isinstance(ao_density, torch.Tensor) and not hasattr(
        ao_density,
        "__cuda_array_interface__",
    ):
        ao_density = np.asanyarray(ao_density)
    if ao_density.ndim == 3:
        if ao_density.shape[0] != 2:
            raise ValueError("spin-resolved ao_density must have two spin channels")
        ao_density = ao_density.sum(axis=0)
    if ao_density.ndim != 2:
        raise ValueError("ao_density must be rank 2 or spin-resolved rank 3")
    return ao_density


def build_projector_molecule(mol, projector_basis):
    """Build the zero-electron molecule that carries atomic projectors."""
    atom_indices = descriptor_atom_indices(mol)
    if not atom_indices:
        raise ValueError("the descriptor requires at least one non-ghost atom")
    coordinates = mol.atom_coords(unit="Angstrom")
    projector_mol = gto.Mole()
    projector_mol.atom = [("X", coordinates[index]) for index in atom_indices]
    projector_mol.basis = projector_basis
    projector_mol.verbose = 0
    projector_mol.build(0, 0, unit="Angstrom")
    return projector_mol


class AtomicDensityDescriptor:
    """Shared projected-density descriptor bound to one PySCF molecule."""

    def __init__(self, mol, projector_basis=None, device=DEFAULT_CUDA_DEVICE):
        self.device = device
        self.projector_basis = deepcopy(load_basis(projector_basis))
        self.shell_sizes = tuple(get_shell_sec(self.projector_basis))
        self.n_features = sum(self.shell_sizes)
        self.mol = None
        self.projector_mol = None
        self.descriptor_atom_indices: tuple[int, ...] = ()
        self.overlap_shells: tuple[torch.Tensor, ...] = ()
        self.reset(mol)

    @property
    def n_descriptor_atoms(self) -> int:
        return len(self.descriptor_atom_indices)

    def reset(self, mol):
        self.mol = mol
        self.descriptor_atom_indices = descriptor_atom_indices(mol)
        self.projector_mol = build_projector_molecule(mol, self.projector_basis)
        overlap = self._tensor(self.projection_overlap())
        self.overlap_shells = tuple(torch.split(overlap, self.shell_sizes, -1))
        self._derivative_overlap_shells = None
        return self

    def _tensor(self, value):
        # Native PySCF spin derivatives contract host AO integrals and densities.
        # Conventional descriptors keep their CUDA runtime checks and DLPack path.
        if torch.device(self.device or DEFAULT_CUDA_DEVICE).type == "cpu":
            return torch.as_tensor(value, dtype=torch.float64, device=self.device)
        return torch_from_array(value, device=self.device)

    def as_ao_density_tensor(self, ao_density) -> torch.Tensor:
        tensor = self._tensor(ao_density)
        if tensor.ndim != 2 or tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("ao_density must be a square rank-2 matrix")
        return tensor

    def cross_integral(self, integral: str) -> np.ndarray:
        return gto.intor_cross(integral, self.mol, self.projector_mol)

    def projection_overlap(self) -> np.ndarray:
        overlap = self.cross_integral("int1e_ovlp")
        return overlap.reshape(
            self.mol.nao,
            self.projector_mol.natm,
            self.projector_mol.nao // self.projector_mol.natm,
        )

    def derivative_overlap_shells(self) -> tuple[torch.Tensor, ...]:
        if self._derivative_overlap_shells is not None:
            return self._derivative_overlap_shells
        derivative_overlap = self._tensor(self.cross_integral("int1e_ipovlp"))
        derivative_overlap = derivative_overlap.reshape(
            3,
            self.mol.nao,
            self.projector_mol.natm,
            -1,
        )
        self._derivative_overlap_shells = tuple(
            torch.split(derivative_overlap, self.shell_sizes, -1)
        )
        return self._derivative_overlap_shells

    def derivative_workspace(self, ao_density, *, operation_hook=None):
        """Build one calculation-scoped owner of reusable derivative primitives."""
        from .workspace import DescriptorDerivativeWorkspace

        return DescriptorDerivativeWorkspace(
            self,
            ao_density,
            operation_hook=operation_hook,
        )

    def torch_projected_density(self, ao_density) -> list[torch.Tensor]:
        return projected_density(
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
        )

    def projected_density(self, ao_density, flatten: bool = False):
        blocks = self.torch_projected_density(ao_density)
        if flatten:
            return torch.cat(
                [block.flatten(-2) for block in blocks],
                dim=-1,
            ).detach().cpu().numpy()
        return [block.detach().cpu().numpy() for block in blocks]

    def torch_descriptor(self, ao_density) -> torch.Tensor:
        return descriptor(
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
        )

    def spin_features(self, ao_density, spin: int) -> torch.Tensor:
        """Project one collinear state into total features and signed shell traces.

        A rank-two closed-shell density carries zero spin. A rank-three density
        is ordered alpha, beta. Each shell trace is invariant to rotations of
        its magnetic components and changes sign under a spin-label exchange.
        """
        if isinstance(ao_density, np.ndarray) and any(s < 0 for s in ao_density.strides):
            # NumPy commonly represents an alpha/beta swap as a reversed view.
            ao_density = ao_density.copy()
        density = torch.as_tensor(ao_density)
        if density.is_complex():
            raise ValueError("spin features require a real collinear density")
        density = density.to(self.overlap_shells[0])
        if density.shape == (self.mol.nao, self.mol.nao) and spin == 0:
            total = self.torch_descriptor(density)
            traces = total.new_zeros((self.n_descriptor_atoms, len(self.shell_sizes)))
        elif density.shape == (2, self.mol.nao, self.mol.nao):
            total = self.torch_descriptor(density[0] + density[1])
            blocks = self.torch_projected_density(density[0] - density[1])
            traces = torch.stack([b.diagonal(dim1=-2, dim2=-1).sum(-1)
                                  for b in blocks], dim=-1)
        else:
            raise ValueError("expected a closed-shell AO matrix or two spin channels")
        global_spin = total.new_full((self.n_descriptor_atoms, 1), abs(spin))
        return torch.cat((total, global_spin, traces), dim=-1)

    def descriptor(self, ao_density) -> np.ndarray:
        return self.torch_descriptor(ao_density).detach().cpu().numpy()

    def dq_dP(self, ao_density) -> np.ndarray:
        result = dq_dP(
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
        )
        return result.detach().cpu().numpy()

    def dD_dR_explicit(self, ao_density, flatten: bool = False):
        blocks = dD_dR_explicit(
            self.mol,
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
        )
        if flatten:
            return torch.cat(
                [block.flatten(-2) for block in blocks],
                dim=-1,
            ).detach().cpu().numpy()
        return [block.detach().cpu().numpy() for block in blocks]

    def dq_dR_explicit(self, ao_density, raw_atom_indices=None) -> np.ndarray:
        result = dq_dR_explicit(
            self.mol,
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
            raw_atom_indices,
        )
        return result.detach().cpu().numpy()

    def dq_dR_explicit_component(
        self,
        ao_density,
        component_density,
        raw_atom_indices=None,
    ) -> np.ndarray:
        """Return one additive component of fixed-density descriptor motion."""
        result = dq_dR_explicit_component(
            self.mol,
            self.as_ao_density_tensor(ao_density),
            self.as_ao_density_tensor(component_density),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
            raw_atom_indices,
        )
        return result.detach().cpu().numpy()

    def correction_derivatives(
        self,
        ao_density,
        motion_density,
        sensitivity,
        raw_atom_indices=None,
    ) -> np.ndarray:
        """Return contracted explicit motion and the AO correction potential."""
        density = self.as_ao_density_tensor(ao_density)
        gradient, potential = contract_descriptor_derivatives(
            self.mol,
            density,
            self.as_ao_density_tensor(motion_density),
            torch.tensor(sensitivity, dtype=density.dtype, device=density.device),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
            raw_atom_indices,
        )
        return (
            gradient.detach().cpu().numpy(),
            potential.detach().cpu().numpy(),
        )
