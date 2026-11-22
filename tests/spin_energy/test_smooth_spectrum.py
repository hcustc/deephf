"""Training and inference semantics of smooth single-state spin checkpoints."""
import numpy as np
import pytest
import torch
from pyscf import gto

from deepks.deephf import SpinTrainingState, make_deephf, spin_energy_force_loss
from deepks.descriptor import DescriptorDifferentiabilityError
from deepks.model import SpinCorrNet
from deepks.model.spin import _smooth_shell
from test_df_spin_forces import BASIS, reference


def model(seed, tau=0.05):
    torch.manual_seed(seed)
    return SpinCorrNet(BASIS, (4,), nominal_spin=False, spectrum_tau=tau,
                       elem_table=[[1, 7, 8], [-.001 * seed, .003 * seed, .002 * seed]]).eval()


def test_smooth_spectrum_repeated_eigenvalues():
    x = torch.tensor([-.2, .1, .1, .10001, .3], dtype=torch.float64, requires_grad=True)
    y = _smooth_shell(x, .01)
    torch.testing.assert_close(y.sum(), x.sum())
    torch.testing.assert_close(_smooth_shell(x.flip(0), .01), y)
    slope, = torch.autograd.grad(y @ torch.arange(5, dtype=x.dtype), x)
    torch.testing.assert_close(slope[1], slope[2], rtol=0, atol=1e-13)


def test_migrate_only_explicit_legacy_checkpoint(tmp_path):
    path = tmp_path / "weights.pth"
    old = model(4, 0)
    old.save(path)
    class ArchivedSpin(SpinCorrNet):
        @classmethod
        def load_dict(cls, data):
            return super().load_dict(data)
    assert ArchivedSpin.load(path).spectrum_tau == 0
    assert SpinCorrNet.load(path).spectrum_tau == 0
    smooth = SpinCorrNet.load(path, legacy_spectrum_tau=.001)
    smooth.save(path)
    loaded = SpinCorrNet.load(path)
    assert loaded.spectrum_tau == .001 and not loaded.nominal_spin
    with pytest.raises(ValueError, match="overridden"):
        SpinCorrNet.load(path, legacy_spectrum_tau=.05)
    for invalid in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="spectrum_tau"):
            model(4, invalid)


def test_degenerate_shell_trains_and_matches_energy_difference():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    ref = reference(0, mol)
    net = model(4)
    state = SpinTrainingState.from_reference(ref, net, ref.e_tot, np.zeros((2, 3)))
    assert state.force_frame.smooth_only
    with pytest.raises(ValueError, match="smooth model"):
        state.force_frame.forces(model(4, 0))
    with pytest.raises(DescriptorDifferentiabilityError):
        make_deephf(ref, model(4, 0)).gradient()
    loss = spin_energy_force_loss(net, [state])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    grad = make_deephf(ref, net).gradient()
    np.testing.assert_allclose(-state.force_frame.forces(net).detach(), grad, atol=1e-9, rtol=0)
    energies = []
    for sign in (1, -1):
        xyz = mol.atom_coords()
        xyz[1, 2] += sign * 1e-4
        moved = mol.set_geom_(xyz, unit="Bohr", inplace=False)
        energies.append(net.total_energy(reference(0, moved, ref.make_rdm1())))
    np.testing.assert_allclose((energies[0]-energies[1])/2e-4, grad[1, 2], atol=2e-7, rtol=0)
