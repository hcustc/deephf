"""Interatomic spin signs, batched states and deployment of a relative-spin model."""

import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.descriptor import AtomicDensityDescriptor
from deepks.model import SpinCorrNet


BASIS = [[0, [0.6, 1.0]], [1, [0.8, 1.0]]]


def two_centers():
    # Five common features and two signed shell traces. Only the traces flip.
    x = torch.tensor(
        [[1, 2, 3, 4, 0, 1, -2], [4, 3, 2, 1, 0, 2, -3]], dtype=torch.float64
    )
    xyz = x.new_tensor([[0, 0, 0], [0, 2, 0]])
    return x, xyz


def test_two_center_relative_spin_symmetry():
    torch.manual_seed(41)
    x, xyz = two_centers()
    model = SpinCorrNet(BASIS, (4,), relative_cutoff_angstrom=4.0)
    flipped = x.clone()
    flipped[0, -2:] *= -1
    swapped = x.clone()
    swapped[:, -2:] *= -1
    c = model.prepare_features(x, xyz.numpy())[:, -2:]
    # The two centers are at half the cutoff, so w = 1/2.
    torch.testing.assert_close(c, x.new_tensor([[1, 3], [1, 3]]))
    torch.testing.assert_close(model.prepare_features(flipped, xyz)[:, -2:], -c)
    torch.testing.assert_close(model(x, xyz), model(swapped, xyz), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        model(x, xyz), model(x.flip([0]), xyz.flip([0])), atol=1e-12, rtol=0
    )
    # Exhibit an exact readout of the first relative channel.
    with torch.no_grad():
        for param in model.network.densenet.parameters():
            param.zero_()
    model.network.set_prefitting([0] * 7 + [1, 0], 0)
    torch.testing.assert_close(
        model(x, xyz) - model(flipped, xyz), x.new_tensor([4 / (1 + 1e-8)])
    )
    for control in (
        SpinCorrNet(BASIS, (4,)),
        SpinCorrNet(BASIS, (4,), relative_cutoff_angstrom=4, mask_relative_spin=True),
    ):
        torch.testing.assert_close(
            control(x, xyz), control(flipped, xyz), atol=1e-12, rtol=0
        )
        if control.mask_relative_spin:
            torch.testing.assert_close(
                control.forward_prepared(model.prepare_features(x, xyz)),
                control(x, xyz),
            )


def test_relative_spin_padding_and_range():
    torch.manual_seed(7)
    x, xyz = two_centers()
    model = SpinCorrNet(BASIS, (4,), relative_cutoff_angstrom=4.0)
    model.network.set_energy_const(0.75)
    for coords in (xyz * 2, xyz * 3):
        assert torch.count_nonzero(model.prepare_features(x, coords)[:, -2:]) == 0
    assert torch.count_nonzero(model.prepare_features(x[:1], xyz[:1])[:, -2:]) == 0
    unpolarized = x.clone()
    unpolarized[:, -2:] = 0
    assert torch.count_nonzero(model.prepare_features(unpolarized, xyz)[:, -2:]) == 0
    padded_x = torch.stack((x, x.clone()))
    padded_xyz = torch.stack((xyz, xyz.clone()))
    mask = torch.tensor([[True, True], [True, False]])
    padded_x[~mask] = float("nan")
    padded_xyz[~mask] = float("nan")
    expected = torch.stack((model(x, xyz), model(x[:1], xyz[:1])))
    torch.testing.assert_close(
        model(padded_x, padded_xyz, mask), expected, atol=1e-12, rtol=0
    )
    with pytest.raises(ValueError, match="positions"):
        model(x)


@pytest.mark.parametrize("masked", [False, True])
def test_relative_checkpoint_separate_states(tmp_path, masked):
    torch.manual_seed(43)
    model = SpinCorrNet(
        BASIS,
        (4,),
        relative_cutoff_angstrom=3.25,
        mask_relative_spin=masked,
        elem_table=[[1], [-0.03]],
    )
    path = tmp_path / "spin.pth"
    model.save(path)
    loaded = SpinCorrNet.load(path)
    assert (
        loaded.relative_cutoff_angstrom == 3.25 and loaded.mask_relative_spin == masked
    )
    for spin in (0, 2):
        # A ghost basis center has neither projected features nor an elemental offset.
        mol = gto.M(
            atom="H 0 0 0; ghost-H 1 0 0; H 0 0 1.5",
            basis="sto-3g",
            spin=spin,
            verbose=0,
        )
        ref = (scf.UHF(mol) if spin else scf.RHF(mol)).run()
        desc = AtomicDensityDescriptor(mol, BASIS, device="cpu")
        x = desc.spin_features(ref.make_rdm1(), spin)
        xyz = torch.from_numpy(mol.atom_coords(unit="Angstrom")[[0, 2]])
        expected = ref.e_tot + model(x, xyz).item() - 0.06
        np.testing.assert_allclose(
            loaded.total_energy(ref), expected, atol=1e-12, rtol=0
        )
