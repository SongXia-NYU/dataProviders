import pandas as pd
import numpy as np
import os
import os.path as osp
import rdkit
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles
from glob import glob


def dataset_distribution(f_paths):
    n_heavy = []
    f_path = None

    for f_path in glob(f_paths):
        dataset = pd.read_csv(f_path)

        for smiles in dataset["SMILES"]:
            mol = MolFromSmiles(smiles)
            n_heavy.append(mol.GetNumHeavyAtoms())

    f_dir = osp.dirname(f_path)
    f_base = osp.basename(f_path).split(".")[0]
    plt.hist(n_heavy, bins=range(min(n_heavy), max(n_heavy)+1))
    plt.xlabel("num of heavy atoms")
    plt.ylabel("count")
    plt.title("dd_sol")
    plt.savefig(osp.join(f_dir, f_base+"_mmff_dist.png"))


if __name__ == '__main__':
    dataset_distribution("../dd_sol_exp/*.csv")
