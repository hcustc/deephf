"""Per-state joint energy/force fitting for local-spin DF-HF corrections."""

import math
from dataclasses import dataclass

import torch
from pyscf.lib import param
from torch.nn import functional as F

from deepks.descriptor import descriptor_atom_indices

from .method import DFSpinDeePHF, SpinForceFrame, _validate_local_model

# Same Hartree and geometry conventions as the archived QM9 fits.
HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class SpinTrainingState:
    """Frozen HF inputs and total-energy/optional force labels in atomic units."""

    features: torch.Tensor
    projector_basis: list
    elements: tuple[int, ...]
    hf_energy: float
    energy_target: float
    force_frame: SpinForceFrame | None
    force_target: torch.Tensor | None

    @classmethod
    def from_reference(cls, reference, model, energy_target, force_target=None, *, device="cpu"):
        """Prepare this state's HF inputs once, before fitting any weights.

        energy_target is a total energy in Hartree. force_target, when present,
        is the negative reference nuclear gradient in Hartree/Bohr.
        """
        _validate_local_model(model)
        attrs = tuple(
            int(reference.mol.atom_charge(i)) for i in descriptor_atom_indices(reference.mol)
        )
        if not math.isfinite(energy_target):
            raise ValueError("the total energy label must be finite")
        frame = None
        if force_target is None:
            # An energy-only state need not pass a force differentiability or
            # orbital-response test. This preserves energy labels at excluded roots.
            descriptor, features = model.state_features(reference)
            basis = descriptor.projector_basis
        else:
            force_target = torch.as_tensor(force_target, dtype=torch.float64).detach().cpu().clone()
            if (
                force_target.shape != (reference.mol.natm, 3)
                or not torch.isfinite(force_target).all()
            ):
                raise ValueError("force labels must be finite with shape (natm, 3)")
            method = DFSpinDeePHF(reference, model, device=device)
            frame = method._force_frame_on_validated_root()
            basis = frame.projector_basis
            features = frame.features
        return cls(
            features,
            basis,
            attrs,
            float(reference.e_tot),
            float(energy_target),
            frame,
            force_target,
        )


def spin_energy_force_loss(model, states, *, force_weight=1.0):
    """Mean state energy Smooth L1 plus the weighted mean of labeled state force losses.

    The Smooth L1 transition is 1 eV for energies and 1 eV/Angstrom for
    Cartesian force components. Each force-labeled state contributes equally
    regardless of atom count. States without force labels still train energy.
    """
    if not math.isfinite(force_weight) or force_weight < 0:
        raise ValueError("force_weight must be finite and nonnegative")
    energy_errors, force_losses = [], []
    for state in states:
        if state.projector_basis != model.network._pbas:
            raise ValueError("training-state projector basis differs from the checkpoint")
        x = state.features.to(model.network.input_shift)
        offset = model.network.get_elem_const(state.elements)
        target = state.energy_target - state.hf_energy - offset
        energy_errors.append((model(x).reshape(()) - target) * HARTREE_TO_EV)
        if state.force_target is not None and force_weight:
            prediction = state.force_frame.forces(model, create_graph=True)
            unit = HARTREE_TO_EV / param.BOHR
            force_losses.append(
                F.smooth_l1_loss(prediction * unit, state.force_target.to(x) * unit)
            )
    if not energy_errors:
        raise ValueError("a minibatch must contain at least one electronic state")
    errors = torch.stack(energy_errors)
    loss = F.smooth_l1_loss(errors, torch.zeros_like(errors))
    if force_losses:
        loss = loss + force_weight * torch.stack(force_losses).mean()
    return loss
