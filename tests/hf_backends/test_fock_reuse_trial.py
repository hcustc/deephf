"""Check the scoped Fock experiment's acceptance and error paths on CPU."""

from contextlib import nullcontext

import numpy as np
from pyscf import gto, scf
import pytest

from deepks.deephf.backends.hf import HFState
from deepks.deephf.df_spin import method
from validation.hf_backends.fock_reuse import OneFrameTrial


@pytest.fixture
def root():
    mol = gto.M(atom="O 0 0 0; H .94 0 0; H -.22 .85 .10", basis="def2-svp",
                spin=1, charge=1, verbose=0)
    new_reference = lambda: scf.UHF(mol).density_fit(auxbasis="def2-svp-jkfit")
    native = new_reference().set(conv_tol=1e-12, conv_tol_grad=1e-8).run()
    assert native.converged
    return HFState(native.e_tot, native.make_rdm1()), new_reference


def test_fock_witness_is_one_shot_and_keeps_external_energy_check(root):
    state, new_reference = root
    trial = OneFrameTrial()
    with trial.enabled():
        ref = state.response_reference(new_reference())
        method._validate_reference(ref)
        assert trial.hits == 1
        method._validate_reference(ref)
        assert trial.hits == 1
        with pytest.raises(ValueError, match="energy is inconsistent"):
            HFState(state.energy + .001, state.density).response_reference(new_reference())
        assert trial.witness is None
    assert "get_veff" not in vars(ref)


@pytest.mark.parametrize("change", ["orbital_energy", "density", "df_factors"])
@pytest.mark.parametrize("reuse", [False, True])
def test_fock_witness_preserves_rejection(root, change, reuse):
    state, new_reference = root
    trial = OneFrameTrial()
    with trial.enabled() if reuse else nullcontext():
        ref = state.response_reference(new_reference())
        if change == "orbital_energy":
            ref.mo_energy[0, 0] += 1e-3
        elif change == "density":
            # Mix an occupied and an empty orbital without changing orthonormality.
            coeff = ref.mo_coeff[0]
            cols = [0, np.flatnonzero(ref.mo_occ[0] == 0)[0]]
            angle = .01
            coeff[:, cols] = coeff[:, cols] @ np.array([
                [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        else:
            ref.with_df._cderi *= 1.001
        with pytest.raises(ValueError, match="canonical stationary"):
            method._validate_reference(ref)
        assert trial.hits == 0
