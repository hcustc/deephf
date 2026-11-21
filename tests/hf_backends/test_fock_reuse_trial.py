"""Check the scoped Fock experiment's acceptance and error paths on CPU."""

from contextlib import nullcontext
from unittest.mock import Mock

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


def test_valid_reference_near_residual_limit_uses_fresh_fock(root, monkeypatch):
    state, new_reference = root
    trial = OneFrameTrial()
    with trial.enabled():
        ref = state.response_reference(new_reference())
        assert trial.witness is not None
        # Put one column inside the production tolerance but outside the reuse margin.
        sc = ref.get_ovlp() @ ref.mo_coeff[0, :, 0]
        ref.mo_energy[0, 0] += 9.5e-8 / np.max(np.abs(sc))
        dm = ref.make_rdm1()
        physical = ref.get_hcore() + ref.get_veff(ref.mol, dm)
        residual = physical @ ref.mo_coeff - (ref.get_ovlp() @ ref.mo_coeff) * ref.mo_energy[:, None, :]
        assert 9e-8 < np.max(np.abs(residual)) < 1e-7

        fresh = Mock(wraps=ref.get_veff)
        monkeypatch.setattr(ref, "get_veff", fresh)
        method._validate_reference(ref)
        fresh.assert_called_once()
        assert trial.hits == 0


@pytest.mark.parametrize("df", [False, True])
def test_rhf_transport_uses_original_fock(df):
    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", verbose=0)
    ref = scf.RHF(mol)
    if df:
        ref = ref.density_fit()
    ref.run()
    trial = OneFrameTrial()
    with trial.enabled():
        HFState(ref.e_tot, ref.make_rdm1()).response_reference(ref)
        assert trial.witness is None
        if df:
            method._validate_reference(ref)


@pytest.mark.parametrize("extra_at", ["capture", "validation"])
def test_incremental_fock_arguments_pass_through(root, monkeypatch, extra_at):
    state, new_reference = root
    ref = new_reference()
    fresh = Mock(wraps=ref.get_veff)
    monkeypatch.setattr(ref, "get_veff", fresh)

    def invoke():
        # Exercise PySCF's optional positional dm_last and keyword hermi.
        return ref.get_veff(ref.mol, state.density, 0, hermi=1)

    if extra_at == "capture":
        monkeypatch.setattr(HFState, "response_reference", lambda *_: invoke())
    else:
        monkeypatch.setattr(method, "_validate_reference", lambda _: invoke())
    trial = OneFrameTrial()
    with trial.enabled():
        state.response_reference(ref)
        if extra_at == "validation":
            assert trial.witness is not None
            method._validate_reference(ref)
        assert trial.witness is None
        assert trial.hits == 0
        fresh.assert_called_with(ref.mol, state.density, 0, hermi=1)
