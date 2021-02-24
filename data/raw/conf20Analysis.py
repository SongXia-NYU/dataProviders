import torch
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from rdkit.Chem import MolToSmiles
from torch_scatter import scatter
import pandas as pd


def conf_dist():
    conf20_csv = pd.read_csv("conf20_analysis/data_consistent_strict_rmrpc.csv")
    conf20_mol = torch.load("conf20_QM.pt")
    conf20_properties = np.load("conf20_QM_PhysNet.npz")
    save_folder = "conf20_analysis"
    if not osp.exists(save_folder):
        os.mkdir(save_folder)
    energies = conf20_properties["E"]
    plt.hist(energies, bins=70)
    plt.xlabel("energy, eV")
    plt.ylabel("count of conformations")
    plt.title("Conf20 dataset energy distribution")
    plt.savefig(osp.join(save_folder, "energy_dist"))
    plt.show()

    conf20_smiles = conf20_csv["QM_SMILES"].values.tolist()
    molecules = list(set(conf20_smiles))
    batch = [molecules.index(i) for i in conf20_smiles]
    batch = torch.as_tensor(batch)
    mean_energies = scatter(torch.as_tensor(energies), batch, dim=0, reduce="mean")
    count = scatter(torch.ones_like(batch), batch, dim=0, reduce="sum")
    plt.scatter(mean_energies, count, alpha=0.1)
    plt.xlabel("mean energy, eV")
    plt.ylabel("count of conformations")
    plt.title("Conf20 mean energy for each molecule")
    plt.savefig(osp.join(save_folder, "energy_mol"))
    plt.show()

    plt.hist(count, bins=500)
    plt.xlabel("Num of Conformations")
    plt.ylabel("count of molecules")
    plt.title("Conf20 number of generated conformation distribution")
    plt.savefig(osp.join(save_folder, "n_conf_dist"))
    plt.xlim([0, 50])
    plt.savefig(osp.join(save_folder, "n_conf_dist_x_lim_small"))
    plt.show()


def compare_crystal():
    save_folder = "conf20_analysis"

    conf20_csv = pd.read_csv("conf20_analysis/data_consistent_strict_rmrpc.csv")
    conf20_mol = torch.load("conf20_QM.pt")
    conf20_properties = np.load("conf20_QM_PhysNet.npz")
    csd20_mol = torch.load("CSD20/CSD20_cry_min_QM.pt")
    csd20_properties = np.load("CSD20/CSD20_PhysNet_QM.npz")

    csd20_smiles = [MolToSmiles(mol, isomericSmiles=False, allBondsExplicit=False, allHsExplicit=False)
                    for mol in csd20_mol]
    conf20_smiles = conf20_csv["QM_SMILES"].values.tolist()
    molecules = list(set(conf20_smiles))
    batch = [molecules.index(i) for i in conf20_smiles]
    batch = torch.as_tensor(batch)

    def _index(source_list: list, i):
        try:
            return source_list.index(i)
        except ValueError:
            return -1

    conf20_in_csd20_index = [_index(csd20_smiles, mol) for mol in molecules]
    cry_energies = torch.as_tensor(csd20_properties["E"])[conf20_in_csd20_index]
    cry_energies_conf = cry_energies[batch]

    energies = torch.as_tensor(conf20_properties["E"])
    mean_energies = scatter(energies, batch, dim=0, reduce="mean")

    energy_diff_mol = torch.abs(cry_energies - mean_energies)
    energy_diff_conf = torch.abs(cry_energies_conf - energies)

    plt.hist(energy_diff_mol)
    plt.xlabel("Energy difference, eV")
    plt.ylabel("Count of Molecules")
    plt.title("Energy distribution by molecules")
    plt.savefig(osp.join(save_folder, "energy_diff_mol"))
    plt.show()

    plt.hist(energy_diff_conf)
    plt.xlabel("Energy difference, eV")
    plt.ylabel("Count of Conformations")
    plt.title("Energy distribution by conformations")
    plt.savefig(osp.join(save_folder, "energy_diff_conf"))


if __name__ == '__main__':
    compare_crystal()
