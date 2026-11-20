"""Train on a neutral/cation pair with one missing force label.

Run from the repository root with `uv run python examples/spin_joint/train.py`.
Labels here come from small PBE calculations to keep the example self-contained.
"""

import torch
from pyscf import dft, gto, lib

from deepks.deephf import SpinTrainingState, build_reference, spin_energy_force_loss
from deepks.model import SpinCorrNet


def main():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    lib.num_threads(1)
    model = SpinCorrNet([[0, [0.5, 1.0]], [1, [0.7, 1.0]]], (8,), nominal_spin=False)
    states = []
    for charge, family in ((0, "rhf"), (1, "uhf")):
        mol = gto.M(
            atom="O 0 0 0; H .94 0 0; H -.23 .85 .13",
            basis="sto-3g",
            charge=charge,
            spin=charge,
            verbose=0,
        )
        hf = build_reference(mol, family, density_fit=True)
        label = (dft.UKS(mol) if charge else dft.RKS(mol)).set(xc="PBE").run()
        if not label.converged:
            raise RuntimeError("reference label calculation did not converge")
        force = -label.nuc_grad_method().kernel() if charge else None
        states.append(SpinTrainingState.from_reference(hf, model, label.e_tot, force))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(3):
        optimizer.zero_grad()
        loss = spin_energy_force_loss(model, states)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch + 1}: joint loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
