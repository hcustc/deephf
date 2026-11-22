"""Mixed DF-Fock derivatives with signed probes and the public GPU4PySCF force path."""

import numpy as np
import pytest
import torch
from pyscf import gto

from deepks.deephf import evaluate_molecule
from deepks.model import SpinCorrNet


@pytest.mark.parametrize("charge", [0, 1])
def test_native_gpu_hf_gradient_in_complete_forces(charge):
    pytest.importorskip("gpu4pyscf")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    mol = gto.M(atom="O 0 0 0; H .95 0 0; H -.24 .91 .04", basis="sto-3g",
                charge=charge, spin=charge, verbose=0)
    family = "uhf" if charge else "rhf"
    aux = "def2-universal-jkfit"
    torch.manual_seed(21)
    net = SpinCorrNet([[0, [0.6, 1.]]], (5,))
    options = dict(family=family, auxbasis=aux, density_fit=True, backend="zvector",
                   scf_args={"conv_tol": 1e-12, "conv_tol_grad": 7e-8})
    base = evaluate_molecule(mol, net, **options)
    result = evaluate_molecule(mol, net, scf_backend="gpu4pyscf", device="cuda",
                               gpu_derivatives="hf", **options)
    np.testing.assert_allclose(result["force"], base["force"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(result["e_tot"], base["e_tot"], rtol=0, atol=1e-8)
