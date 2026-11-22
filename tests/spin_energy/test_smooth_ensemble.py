"""Checkpoint semantics, shell derivatives and shared-root ensemble force regression."""

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from deepks.deephf import SpinDeePHFCalculator, make_deephf
from deepks.deephf.df_spin.orbitals import DFOrbitalResponse
from deepks.model import SpinEnsemble
from test_smooth_spectrum import model
from test_df_spin_forces import reference


def test_mixed_spectra_rejected():
    with pytest.raises(ValueError, match="feature mapping"):
        SpinEnsemble([model(4), model(4, 0)])


@pytest.mark.parametrize("charge", [0, 1])
def test_shared_adjoint_and_cartesian_derivative(charge):
    ref = reference(charge)
    members = [model(3), model(13), model(20)]
    singles = [make_deephf(ref, m) for m in members]
    energy = np.mean([m.kernel() for m in singles])
    gradient = np.mean([m.gradient() for m in singles], axis=0)
    ensemble = SpinEnsemble(members)
    combined = make_deephf(ref, ensemble)
    with patch.object(DFOrbitalResponse, "adjoint", autospec=True,
                      side_effect=DFOrbitalResponse.adjoint) as solve:
        g = combined.gradient()
    assert solve.call_count == 1
    np.testing.assert_allclose(combined.kernel(), energy, rtol=0, atol=2e-13)
    np.testing.assert_allclose(g, gradient, rtol=0, atol=1e-9)
    np.testing.assert_allclose(combined.gradient(backend="direct"), g, rtol=0, atol=1e-9)
    xyz = ref.mol.atom_coords()
    energies = []
    for sign in (1, -1):
        other = xyz.copy()
        other[1, 0] += sign * 1e-4
        displaced = ref.mol.set_geom_(other, unit="Bohr", inplace=False)
        energies.append(ensemble.total_energy(reference(charge, displaced, ref.make_rdm1())))
    np.testing.assert_allclose((energies[0]-energies[1])/2e-4, g[1, 0], rtol=0, atol=2e-7)


def test_ase_keeps_last_successful_frame(monkeypatch):
    ref = reference(0)
    atoms = Atoms(numbers=ref.mol.atom_charges(), positions=ref.mol.atom_coords(unit="Angstrom"))
    net = SpinEnsemble([model(4), model(7)])
    calc = SpinDeePHFCalculator(ref.mol, net, family="rhf", scf_args={"conv_tol": 1e-13})
    atoms.calc = calc
    expected = make_deephf(ref, net)
    np.testing.assert_allclose(atoms.get_forces(), expected.forces()*Hartree/Bohr, atol=3e-6, rtol=0)
    np.testing.assert_allclose(atoms.get_potential_energy(), expected.kernel()*Hartree, atol=1e-7, rtol=0)
    assert len(calc.sequence.records) == 1
    atoms.positions[1, 0] += .001
    from ase.calculators.calculator import CalculationFailed
    with monkeypatch.context() as scope:
        def fail(ref):
            raise RuntimeError("failed force")
        scope.setattr(calc, "_evaluate", fail)
        with pytest.raises(CalculationFailed, match="failed force"):
            atoms.get_forces()
    assert len(calc.sequence.records) == 1
    atoms.get_forces()
    assert calc.sequence.records[-1]["initial_guess_source"] == "previous_reference"
