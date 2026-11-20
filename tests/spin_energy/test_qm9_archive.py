"""Replay held-out QM9 predictions with the public DF-HF method on CPU."""

from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.lib import param

from deepks.deephf import make_deephf
from deepks.model import SpinCorrNet


ARCHIVE = Path(__file__).with_name("data") / "qm9_methanol"
HARTREE_TO_EV = 27.211386245988


@pytest.fixture(scope="module", params=[0, 1], ids=["neutral", "cation"])
def archived_state(request):
    charge = request.param
    with np.load(ARCHIVE / "predictions.npz") as data:
        expected = dict(data)
    mol = gto.M(
        atom=list(zip(expected["z"], expected["xyz"])),
        unit="Angstrom", basis="def2-svp", charge=charge, spin=charge, verbose=0,
    )
    reference = (scf.UHF(mol) if charge else scf.RHF(mol)).density_fit()
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-8
    reference.max_cycle = 150
    reference.kernel()
    assert reference.converged
    np.testing.assert_allclose(
        reference.e_tot, expected["hf_energy_hartree"][charge], rtol=0, atol=1e-9
    )
    return charge, reference, expected


@pytest.mark.parametrize("checkpoint", ["energy_A", "energy_B", "joint_A", "joint_B"])
def test_archived_state_energy_and_cartesian_force(archived_state, checkpoint):
    charge, reference, expected = archived_state
    model = SpinCorrNet.load(ARCHIVE / f"{checkpoint}.pth")
    method = make_deephf(reference, model)
    energy = method.kernel()
    force = -method.gradient() * (HARTREE_TO_EV / param.BOHR)
    # Archived energies use the original HF densities. Forces use tightened
    # stationary roots, and joint training stored the geometry Jacobian in FP32.
    np.testing.assert_allclose(
        energy, expected[checkpoint + "_total_hartree"][charge],
        rtol=0, atol=1e-4 / HARTREE_TO_EV,
    )
    np.testing.assert_allclose(
        force, expected[checkpoint + "_force_ev_per_angstrom"][charge],
        rtol=0, atol=2e-5,
    )
