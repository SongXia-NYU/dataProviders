import argparse

import torch
import os.path as osp

import torch_geometric

from DummyIMDataset import DummyIMDataset
from ase.atoms import Atoms
from dscribe.descriptors.acsf import ACSF
from tqdm import tqdm


def add_acsf(dataset_name, save_name):
    acsf = ACSF(
        species=['B', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S'],
        rcut=6.0,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]], )

    dataset = DummyIMDataset(root="../data", dataset_name=dataset_name)
    data_list = []

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        this_data = dataset[i]
        atoms = Atoms(numbers=this_data.Z.numpy(), positions=this_data.R.numpy())
        this_acsf = torch.as_tensor(acsf.create(atoms, n_jobs=1))
        this_data.acsf = this_acsf

        data_list.append(this_data)

    del dataset

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), osp.join("../data/processed", save_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name")
    parser.add_argument("--save_name")
    args = parser.parse_args()
    args = vars(args)

    add_acsf(**args)


if __name__ == '__main__':
    main()
