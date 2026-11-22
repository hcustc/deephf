"""Spin-resolved, single-state DeePHF energy corrections."""

import math
from functools import lru_cache
from itertools import permutations

import torch
from pyscf import scf

from deepks.descriptor import AtomicDensityDescriptor, descriptor_atom_indices
from deepks.model.model import CorrNet
from deepks.utils import get_shell_sec, load_basis


def _validate_atom_mask(features, atom_mask):
    if atom_mask is not None and (
        not isinstance(atom_mask, torch.Tensor)
        or atom_mask.dtype != torch.bool
        or atom_mask.shape != features.shape[:-1]
    ):
        raise ValueError("atom_mask must be boolean with one entry per atom")


@lru_cache(maxsize=16)
def _shell_orders(width, device):
    return torch.tensor(list(permutations(range(width))), dtype=torch.long, device=device)


def _smooth_shell(values, tau):
    """Average rank slots over permutations. Subtract the ordered logit for stability."""
    width = values.shape[-1]
    candidates = values[..., _shell_orders(width, values.device)]
    ranks = torch.arange(width, device=values.device, dtype=values.dtype)
    scores = ((candidates - values.unsqueeze(-2)) * ranks).sum(-1) / tau
    return (scores.softmax(-1).unsqueeze(-1) * candidates).sum(-2)


class SpinEnergyModel(torch.nn.Module):
    """Shared reference, descriptor and elemental-energy conventions for spin inference."""

    def state_features(self, reference):
        """Build this checkpoint's projector and features for a converged collinear HF state."""
        native = reference.undo_df() if hasattr(reference, "undo_df") else reference
        if type(native) not in (scf.hf.RHF, scf.uhf.UHF):
            raise TypeError("SpinCorrNet expects a native RHF or UHF reference")
        if not reference.converged:
            raise ValueError("the single-state reference must be converged")
        descriptor = AtomicDensityDescriptor(reference.mol, self.projector_basis, device="cpu")
        ao_density = reference.make_rdm1()
        features = descriptor.spin_features(ao_density, reference.mol.spin)
        return descriptor, features

    @torch.no_grad()
    def correction_energy(self, reference):
        """Return the single-state correction in Hartree, including elemental offsets."""
        descriptor, features = self.state_features(reference)
        return self._correction(descriptor, features)

    @torch.no_grad()
    def correction_from_density(self, molecule, density):
        """Evaluate the frozen network on a converged RHF or alpha/beta AO density."""
        descriptor = AtomicDensityDescriptor(molecule, self.projector_basis, device="cpu")
        features = descriptor.spin_features(density, molecule.spin)
        return self._correction(descriptor, features)

    def _correction(self, descriptor, features):
        positions = None
        if self.relative_cutoff_angstrom is not None:
            xyz = descriptor.mol.atom_coords(unit="Angstrom")
            positions = torch.as_tensor(xyz[list(descriptor.descriptor_atom_indices)])
        energy = self(features.to(self.input_reference), positions)
        return self.correction_with_offsets(descriptor.mol, energy)

    def correction_with_offsets(self, molecule, energy):
        """Add elemental baselines to a scalar network energy and return Hartree."""
        elements = [int(molecule.atom_charge(i)) for i in descriptor_atom_indices(molecule)]
        if isinstance(energy, torch.Tensor):
            energy = energy.item()
        correction = float(energy) + self.element_offset(elements)
        if not math.isfinite(correction):
            raise ValueError("single-state energy correction must be finite")
        return float(correction)

    def total_energy(self, reference):
        """Compose the HF and learned energies for this single electronic state."""
        correction = self.correction_energy(reference)
        return float(reference.e_tot) + correction


class SpinCorrNet(SpinEnergyModel):
    """An atomic CorrNet averaged over the two equivalent spin labelings.

    Inputs end in [total-density eigenvalues, |Nα-Nβ|, shell spin traces].
    The same nonlinear network sees both signs of the local spin vector,
    preserving relative signs across shells while making energy invariant to
    interchanging alpha and beta. The global-only control has the same architecture
    and dimensions but masks all local spin channels. A positive neighborhood
    cutoff adds same-shell interatomic spin products to the even channels.
    """

    NO_LABEL_FORMAT = "qm9-ie.no-nominal-spin.v1"
    SMOOTH_FORMAT = "deepks.spin-energy.v3"
    CHECKPOINT_FORMATS = ("deepks.spin-energy.v1", "deepks.spin-energy.v2", NO_LABEL_FORMAT, SMOOTH_FORMAT)

    def __init__(
        self,
        proj_basis=None,
        hidden_sizes=(64, 64),
        *,
        local_spin=True,
        nominal_spin=True,
        elem_table=None,
        relative_cutoff_angstrom=None,
        mask_relative_spin=False,
        spectrum_tau=0.0,
    ):
        super().__init__()
        basis = load_basis(proj_basis)
        shells = get_shell_sec(basis)
        self.shell_sizes = tuple(shells)
        self.spectrum_tau = float(spectrum_tau)
        if not math.isfinite(self.spectrum_tau) or self.spectrum_tau < 0:
            raise ValueError("spectrum_tau must be finite and nonnegative")
        if self.spectrum_tau and max(shells) > 5:
            raise ValueError("permutation smoothing supports s, p and d projector shells")
        self.n_spin = len(shells)
        self.n_common = sum(shells) + 1
        self.local_spin = local_spin
        self.nominal_spin = nominal_spin
        self.relative_cutoff_angstrom = relative_cutoff_angstrom
        self.mask_relative_spin = mask_relative_spin
        if relative_cutoff_angstrom is not None and (
            not local_spin
            or not nominal_spin
            or not math.isfinite(relative_cutoff_angstrom)
            or relative_cutoff_angstrom <= 0
        ):
            raise ValueError(
                "relative spin requires local spin, nominal spin and a finite positive cutoff"
            )
        if mask_relative_spin and relative_cutoff_angstrom is None:
            raise ValueError("mask_relative_spin requires a relative spin cutoff")
        n_relative = self.n_spin if relative_cutoff_angstrom is not None else 0
        self.network = CorrNet(
            self.n_common + self.n_spin + n_relative,
            hidden_sizes=hidden_sizes,
            proj_basis=basis,
            elem_table=elem_table,
        )

    @property
    def projector_basis(self):
        return self.network._pbas

    @property
    def input_reference(self):
        return self.network.input_shift

    def element_offset(self, elements):
        return self.network.get_elem_const(elements)

    def prepare_features(self, features, positions=None, atom_mask=None):
        """Build cacheable atomic inputs for whole states before any atom splitting.

        Positions are in Angstrom, in descriptor atom order. A boolean mask of
        shape (..., atoms) identifies real centers in a padded state batch.
        """
        if features.ndim < 2 or features.shape[-1] != self.n_common + self.n_spin:
            raise ValueError("expected projected spin features with an atom axis")
        _validate_atom_mask(features, atom_mask)
        if atom_mask is not None:
            features = features.masked_fill(~atom_mask.unsqueeze(-1), 0)
        if self.spectrum_tau:
            spectrum, spin_channels = features.split([self.n_common - 1, self.n_spin + 1], dim=-1)
            shells = spectrum.split(self.shell_sizes, dim=-1)
            features = torch.cat([
                *(_smooth_shell(s, self.spectrum_tau) if s.shape[-1] > 1 else s for s in shells),
                spin_channels,
            ], dim=-1)
        if self.relative_cutoff_angstrom is None:
            return features
        if positions is not None:
            positions = torch.as_tensor(
                positions, device=features.device, dtype=features.dtype
            )
        if positions is None or positions.shape != (*features.shape[:-1], 3):
            raise ValueError(
                "relative spin requires positions in Angstrom for each atom"
            )
        spin = features[..., self.n_common :]
        if self.mask_relative_spin:
            relative = torch.zeros_like(spin)
        else:
            if atom_mask is not None:
                positions = positions.masked_fill(~atom_mask.unsqueeze(-1), 0)
            distances = torch.cdist(
                positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
            )
            phase = (distances / self.relative_cutoff_angstrom).clamp(max=1) * math.pi
            weights = (phase.cos() + 1) * 0.5
            # Remove self interactions by index, also for coincident centers.
            diagonal = torch.eye(spin.shape[-2], dtype=torch.bool, device=spin.device)
            relative = spin * (weights.masked_fill(diagonal, 0) @ spin)
        return torch.cat((features, relative), dim=-1)

    def forward_prepared(self, features, atom_mask=None):
        """Sum atomic corrections from prepare_features, excluding elemental offsets."""
        _validate_atom_mask(features, atom_mask)
        spin_end = self.n_common + self.n_spin
        common, spin, relative = torch.split(
            features,
            [self.n_common, self.n_spin, self.network.input_dim - spin_end],
            dim=-1,
        )
        if not self.nominal_spin:
            common = torch.cat((common[..., :-1], torch.zeros_like(common[..., -1:])), dim=-1)
            features = torch.cat((common, spin, relative), dim=-1)
        if not self.local_spin:
            return self.network(
                torch.cat((common, torch.zeros_like(spin)), dim=-1), atom_mask
            )
        if self.mask_relative_spin:
            # Also mask a physical descriptor cache shared with the unmasked model.
            relative = torch.zeros_like(relative)
            features = torch.cat((common, spin, relative), dim=-1)
        # A global alpha/beta exchange negates the local traces only.
        reverse = torch.cat((common, -spin, relative), dim=-1)
        return (
            self.network(features, atom_mask) + self.network(reverse, atom_mask)
        ) * 0.5

    def forward(self, features, positions=None, atom_mask=None):
        """Evaluate each electronic state separately, returning energies in Hartree."""
        return self.forward_prepared(
            self.prepare_features(features, positions, atom_mask), atom_mask
        )

    def save(self, filename):
        """Write a versioned energy checkpoint including projector and offsets."""
        data = dict(
            format=self.CHECKPOINT_FORMATS[0] if self.nominal_spin else self.NO_LABEL_FORMAT,
            local_spin=self.local_spin,
            network=self.network.save_dict(),
        )
        if self.relative_cutoff_angstrom is not None:
            data.update(
                format=self.CHECKPOINT_FORMATS[1],
                relative_cutoff_angstrom=self.relative_cutoff_angstrom,
                mask_relative_spin=self.mask_relative_spin,
            )
        if self.spectrum_tau:
            data.update(format=self.SMOOTH_FORMAT, spectrum_tau=self.spectrum_tau,
                        nominal_spin=self.nominal_spin)
        torch.save(data, filename)

    @classmethod
    def load(cls, filename, *, legacy_spectrum_tau=None):
        """Restore a spin-energy checkpoint on CPU for single-state inference."""
        data = torch.load(filename, map_location="cpu", weights_only=True)
        if legacy_spectrum_tau is None:
            return cls.load_dict(data)
        return cls.load_dict(data, legacy_spectrum_tau=legacy_spectrum_tau)

    @classmethod
    def load_dict(cls, data, *, legacy_spectrum_tau=None):
        """Restore the architecture and its spin semantics from a checkpoint."""
        if data.get("format") not in cls.CHECKPOINT_FORMATS:
            raise ValueError("expected a versioned spin-energy checkpoint")
        if data["format"] == cls.SMOOTH_FORMAT:
            if legacy_spectrum_tau is not None:
                raise ValueError("a versioned spectrum cannot be overridden")
            tau = data["spectrum_tau"]
            if not tau > 0:
                raise ValueError("a smooth checkpoint requires positive spectrum_tau")
            nominal = data["nominal_spin"]
            has_relative = "relative_cutoff_angstrom" in data
        else:
            # Legacy smooth retraining used an external transform; only its
            # owner can identify those weights. All other legacy files remain hard.
            tau = 0.0 if legacy_spectrum_tau is None else legacy_spectrum_tau
            nominal = data["format"] != cls.NO_LABEL_FORMAT
            has_relative = data["format"] == cls.CHECKPOINT_FORMATS[1]
        net = data["network"]
        args = net["init_args"]
        model = cls(
            args["proj_basis"],
            args["hidden_sizes"],
            local_spin=data["local_spin"],
            nominal_spin=nominal,
            spectrum_tau=tau,
            elem_table=args["elem_table"],
            relative_cutoff_angstrom=(
                data["relative_cutoff_angstrom"] if has_relative else None
            ),
            mask_relative_spin=data["mask_relative_spin"] if has_relative else False,
        )
        model.network.load_state_dict(net["state_dict"], strict=True)
        return model.eval()

class SpinEnsemble(SpinEnergyModel):
    """Equally weighted corrections on one HF root, with one shared descriptor and adjoint."""

    def __init__(self, members):
        super().__init__()
        members = list(members)
        if not members or not all(type(m) is SpinCorrNet for m in members):
            raise ValueError("a spin ensemble requires one or more native SpinCorrNet checkpoints")
        first = members[0]
        for member in members[1:]:
            if (member.projector_basis != first.projector_basis
                    or member.spectrum_tau != first.spectrum_tau
                    or member.relative_cutoff_angstrom != first.relative_cutoff_angstrom
                    or member.mask_relative_spin != first.mask_relative_spin):
                raise ValueError("ensemble members must share the projector and feature mapping")
            if (member.input_reference.dtype != first.input_reference.dtype
                    or member.input_reference.device != first.input_reference.device):
                raise ValueError("ensemble members must use the same network precision and device")
        self.members = torch.nn.ModuleList(members)

    @property
    def projector_basis(self):
        return self.members[0].projector_basis

    @property
    def spectrum_tau(self):
        return self.members[0].spectrum_tau

    @property
    def input_reference(self):
        return self.members[0].input_reference

    @property
    def relative_cutoff_angstrom(self):
        return self.members[0].relative_cutoff_angstrom

    def element_offset(self, elements):
        return sum(m.element_offset(elements) for m in self.members) / len(self.members)

    def forward(self, features, positions=None, atom_mask=None):
        prepared = self.members[0].prepare_features(features, positions, atom_mask)
        energies = [m.forward_prepared(prepared, atom_mask) for m in self.members]
        return torch.stack(energies).mean(0)
