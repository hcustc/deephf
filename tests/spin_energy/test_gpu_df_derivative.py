"""Mixed DF-Fock derivatives with signed probes and the public GPU4PySCF force path."""

import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import evaluate_molecule
from deepks.deephf.df_spin.integrals import DFFockDerivative
from deepks.model import SpinCorrNet


@pytest.mark.parametrize("charge", [0, 1])
def test_gpu_mixed_fock_and_complete_forces(charge):
    pytest.importorskip("gpu4pyscf")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from deepks.deephf.df_spin.integrals import gpu_fock_derivative

    mol = gto.M(atom="O 0 0 0; H .95 0 0; H -.24 .91 .04", basis="sto-3g",
                charge=charge, spin=charge, verbose=0)
    family = "uhf" if charge else "rhf"
    aux = "def2-universal-jkfit"
    ref = (scf.UHF(mol) if charge else scf.RHF(mol)).density_fit(auxbasis=aux).run(conv_tol=1e-13)
    native = ref.to_gpu()
    native.with_df.build()
    cpu = DFFockDerivative(ref)
    q = np.random.default_rng(93).normal(size=(1+charge, mol.nao, mol.nao))
    q = (q + q.swapaxes(-1, -2)) * .5
    for scale in (1., 1e-7, 0.):
        np.testing.assert_allclose(gpu_fock_derivative(ref, native, scale*q),
                                   cpu.contract(scale*q), atol=1e-9*scale, rtol=1e-10)
    # One polarization can vanish identically for a probe parallel to the density.
    density_probe = ref.make_rdm1().reshape(q.shape)
    np.testing.assert_allclose(gpu_fock_derivative(ref, native, density_probe),
                               cpu.contract(density_probe), atol=1e-9, rtol=1e-10)
    torch.manual_seed(21)
    net = SpinCorrNet([[0, [0.6, 1.]]], (5,))
    options = dict(family=family, auxbasis=aux, density_fit=True, backend="zvector",
                   scf_args={"conv_tol": 1e-12, "conv_tol_grad": 7e-8})
    base = evaluate_molecule(mol, net, **options)
    for mode in ("hf", "hf+df"):
        result = evaluate_molecule(mol, net, scf_backend="gpu4pyscf", device="cuda",
                                   gpu_derivatives=mode, **options)
        np.testing.assert_allclose(result["force"], base["force"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(result["e_tot"], base["e_tot"], rtol=0, atol=1e-8)
