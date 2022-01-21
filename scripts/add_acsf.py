import torch
import os.path as osp

import torch_geometric

from DummyIMDataset import DummyIMDataset
from ase.atoms import Atoms
from dscribe.descriptors.acsf import ACSF
from tqdm import tqdm


def add_acsf():
    dataset_name = "frag20reducedAllSolRef-Bmsg-cutoff-10.00-sorted-defined_edge-lr-MMFF.pt"
    save_name = "frag20reducedAllSolRef-Bmsg-cutoff-10.00-sorted-defined_edge-lr-acsf308-MMFF.pt"

    acsf = ACSF(species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S', 'I'], rcut=6.0,
                g2_params=[[1, 1], [1, 2], [1, 3]], g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]])

    dataset = DummyIMDataset(root="../data", dataset_name=dataset_name)
    data_list = []

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        this_data = dataset[i]
        atoms = Atoms(numbers=this_data.Z.numpy(), positions=this_data.R.numpy())
        this_acsf = acsf.create(atoms, n_jobs=4)
        this_data.acsf = this_acsf

        data_list.append(this_data)

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), osp.join("../data/processed", save_name))


if __name__ == '__main__':
    add_acsf()
