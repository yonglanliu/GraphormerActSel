# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.


from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from collections import defaultdict

def murcko_scaffold_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return None
    return Chem.MolToSmiles(scaf, isomericSmiles=False)

def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1, seed=123):
    rng = np.random.default_rng(seed)

    scaff2idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = murcko_scaffold_smiles(smi)
        scaff2idx[scaf].append(i)

    # sort scaffold groups by size (largest first) – common practice
    scaffold_groups = list(scaff2idx.values())
    scaffold_groups.sort(key=len, reverse=True)

    n = len(smiles_list)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)

    train_idx, valid_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx += group
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx += group
        else:
            test_idx += group

    # optional shuffle within each split for batching
    rng.shuffle(train_idx); rng.shuffle(valid_idx); rng.shuffle(test_idx)
    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)
