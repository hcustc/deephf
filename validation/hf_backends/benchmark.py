"""Time complete spin-model energy predictions on one Slurm-allocated GPU.

Run each backend in a fresh process. Each size starts from a fresh SCF guess
even on repetitions. The first call may pay one-time CUDA initialization costs.
"""

import argparse
import csv
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

from pyscf import gto, lib
import torch

from deepks.deephf import evaluate_molecule
from deepks.model import SpinCorrNet


def water_cluster(count):
    atoms = []
    for i in range(count):
        x, y, z = (3.3 * (i % 2), 3.3 * ((i // 2) % 2), 3.3 * (i // 4))
        atoms.extend([("O", (x, y, z)), ("H", (x + .96, y, z)),
                      ("H", (x - .24, y + .93, z))])
    return gto.M(atom=atoms, unit="Angstrom", basis="def2-svp", verbose=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("pyscf", "gpu4pyscf", "generativeqc"))
    parser.add_argument("--waters", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.waters + [args.repeats, args.threads]) < 1:
        parser.error("counts must be positive")
    lib.num_threads(args.threads)
    torch.set_num_threads(args.threads)
    fixture = Path(__file__).resolve().parents[2] / "tests/spin_energy/data/qm9_methanol"
    model = SpinCorrNet.load(fixture / "joint_B.pth")
    writer = csv.writer(sys.stdout)
    writer.writerow(["engine", "n_water", "nao", "first_s", "repeat_median_s", "e_tot_hartree"])
    for count in args.waters:
        mol = water_cluster(count)
        timings = []
        for _ in range(args.repeats + 1):
            start = perf_counter()
            result = evaluate_molecule(mol, model, family="rhf", density_fit=True,
                                       scf_backend=args.engine, energy_only=True)
            timings.append(perf_counter() - start)
        writer.writerow([args.engine, count, mol.nao, timings[0], median(timings[1:]),
                         float(result["e_tot"])])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
