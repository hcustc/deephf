# Held-out methanol regression data

The four checkpoints are original fold 1, seed 41 models from the fixed-geometry QM9 ionization experiment. Model A masks both the nominal spin and local spin traces. Model B retains the local traces. Each checkpoint was selected with methanol's formula held out. The regression uses the single initial geometry of `dsgdb9nsd_000008` (row 7, geometry SHA-256 `bafd5b20f54b02468dfb0d7c884d493072a46755782d99e601e79df3239226fa`).

| Fixture | Archive source |
| --- | --- |
| `energy_{A,B}.pth` | `qm9_ie_spin_no_label_20261120/fold_1/seed_41/{A,B}.pth` |
| `joint_{A,B}.pth` | `qm9_ie_joint_20261120/fold_1/seed_41/{A,B}.pth` |
| Geometry and bare HF energies | `qm9_ie_spin_v1/pairs.json` and `hf/00007.npz` |
| Energy-only total energies and forces | State reference energies from `pairs.json` plus signed prediction errors from the checkpoint's sibling CSV, and Cartesian forces from `qm9_ie_force_20261120/predictions/011.npz` |
| Joint total energies and forces | State reference energies plus signed errors from the checkpoint's sibling NPZ (`index`, `error`), and forces from its `force_index` and `force` arrays |

`predictions.npz` stores neutral then cation energies in Hartree and unpadded Cartesian forces in eV/Å. The archive conversion is 27.211386245988 eV/Hartree. The expected values come from predictions saved before integration.
