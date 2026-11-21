"""CPU checks of the public HF state, solver configuration and warm-start contracts."""

from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pyscf import dft, gto, scf

from deepks.deephf import HFState, build_hf_state, build_reference, workflow
from deepks.deephf.backends import hf


@pytest.mark.parametrize("family,charge,spin", [("rhf", 0, 0), ("uhf", 1, -1)])
def test_public_hf_state_preserves_spin_counts(family, charge, spin):
    mol = gto.M(atom="O 0 0 0; H .94 0 0; H -.23 .85 .13", charge=charge, spin=spin,
                basis="sto-3g", verbose=0)
    ref = (scf.RHF(mol) if family == "rhf" else scf.UHF(mol)).run(conv_tol=1e-13)
    state = build_hf_state(mol, family)
    assert isinstance(state, HFState)
    state.validate(mol, family == "rhf", ref.get_ovlp())
    np.testing.assert_allclose(state.energy, ref.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(state.density, ref.make_rdm1(), atol=2e-7, rtol=0)
    if spin < 0:
        # The native warm-state ABI puts the majority (beta here) block first.
        native = hf._spin_order(state.density, spin)
        np.testing.assert_array_equal(native[0], state.density[1])
        np.testing.assert_array_equal(hf._spin_order(native, spin), state.density)


@pytest.mark.parametrize("density,error", [
    ([[2., 0.1], [0., 0.]], "not symmetric"),
    ([[1., 0.], [0., 1.]], "not an occupied projector"),
    ([[1., 0.], [0., 0.]], "spin counts"),
    ([[np.nan, 0.], [0., 0.]], "invalid external HF"),
])
def test_invalid_density(density, error):
    mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
    # Identity overlap isolates each invariant in an orthonormal test basis.
    with pytest.raises(ValueError, match=error):
        HFState(0., np.asarray(density)).validate(mol, True, np.eye(2))


@pytest.mark.parametrize("factory", [scf.ROHF, dft.RKS], ids=["rohf", "rks"])
@pytest.mark.parametrize("density_fit", [False, True], ids=["exact", "df"])
def test_transport_and_warm_start_reject_other_methods(factory, density_fit):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.77", basis="sto-3g", verbose=0)
    ref = factory(mol)
    if density_fit:
        ref = ref.density_fit()
    ref.kernel()
    state = HFState(ref.e_tot, ref.make_rdm1())
    with pytest.raises(TypeError, match="RHF or UHF"):
        state.response_reference(ref)
    with pytest.raises(ValueError, match="converged molecular HF reference"):
        build_hf_state(mol, "rhf", density_fit=density_fit, initial_reference=ref)


def test_seed_and_density_guess_are_mutually_exclusive():
    mol = gto.M(atom="H 0 0 0; H 0 0 .77", verbose=0)
    ref = scf.RHF(mol).run()
    with pytest.raises(ValueError, match="only one of dm0 and initial_reference"):
        build_reference(mol, "rhf", scf_backend="pyscf", dm0=ref.make_rdm1(), initial_reference=ref)


@pytest.mark.parametrize("family,fit,override,expected", [
    ("rhf", False, None, 1e-10), ("uhf", False, None, 1e-10),
    ("rhf", True, None, 1e-8), ("uhf", True, None, 5e-8),
    ("uhf", True, 2e-9, 2e-9), ("rhf", False, 3e-9, 3e-9),
])
def test_gpu_adapter_passes_gradient_tolerance(monkeypatch, family, fit, override, expected):
    mol = gto.M(atom="H 0 0 0; H 0 0 .77", verbose=0)
    # Exercise the adapter boundary with a NumPy-returning driver on a CPU.
    driver = (scf.RHF(mol) if family == "rhf" else scf.UHF(mol)).run()
    gpu = SimpleNamespace(with_df=SimpleNamespace(auxbasis="weigend") if fit else None,
                          e_tot=driver.e_tot, make_rdm1=driver.make_rdm1, converged=True)

    def kernel(dm0):
        assert gpu.conv_tol_grad == expected

    gpu.kernel = kernel
    monkeypatch.setitem(sys.modules, "gpu4pyscf", ModuleType("gpu4pyscf"))
    monkeypatch.setattr(hf, "require_cuda_device", lambda: None)
    source = driver.density_fit(auxbasis="weigend") if fit else driver
    monkeypatch.setattr(type(source), "to_gpu", lambda self: gpu)
    args = {} if override is None else {"conv_tol_grad": override}
    state = build_hf_state(mol, family, density_fit=fit, auxbasis="weigend" if fit else None,
                           scf_backend="gpu4pyscf", scf_args=args)
    np.testing.assert_array_equal(state.density, driver.make_rdm1())


def test_incompatible_generativeqc_batch_reports_warm_state_abi(monkeypatch):
    mol = gto.M(atom="H 0 0 0; H 0 0 .77", verbose=0)
    ref = scf.RHF(mol).run()
    native = ModuleType("generativeqc._native")
    native.HfWarmState, native.ABI_VERSION, native.check = object, 1, None
    monkeypatch.setitem(sys.modules, "generativeqc._native", native)
    calc = SimpleNamespace(prepare_batch=lambda *a, **kw: nullcontext(object()))
    with pytest.raises(RuntimeError, match="HF warm-state ABI") as error:
        hf._seeded_generativeqc(calc, mol, ref)
    assert isinstance(error.value.__cause__, AttributeError)


def test_cli_records_resolved_automatic_fitting_basis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "h2.xyz").write_text("2\n\nH 0 0 0\nH .77 0 0\n")
    [(directory, _)] = workflow.main(["h2.xyz"], reference="rhf", density_fit=True,
                                    energy_only=True, dump_dir="output")
    payload = json.loads((tmp_path / directory / workflow.INFERENCE_PROVENANCE_FILENAME).read_text())
    mol = gto.M(atom="H 0 0 0; H .77 0 0", basis="sto-3g", verbose=0)
    expected = scf.RHF(mol).density_fit().run().with_df.auxmol.basis
    assert payload["scf_backend"] == "pyscf"
    assert payload["scf_device"] == "cpu"
    assert payload["density_fit"] is True
    assert payload["auxbasis"] == expected
