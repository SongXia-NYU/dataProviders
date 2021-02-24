import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from rdkit import Chem
from torch_scatter import scatter
from tqdm import tqdm


def _dist_from_r_edge(r, edge):
    point1 = r[edge[0, :], :]
    point2 = r[edge[1, :], :]
    dist = torch.sum((point1-point2)**2, dim=-1, keepdim=True)
    dist = torch.sqrt(dist)
    return dist.view(-1)


def view_frag20_data():
    """
    Frag 20 data:
    raw/Frag20_20_sdf_QM.pt:   10207
    raw/Frag20_10_sdf_QM.pt:   143180
    raw/Frag20_11_sdf_QM.pt:   17269
    raw/Frag20_12_sdf_QM.pt:   21502
    raw/Frag20_13_sdf_QM.pt:   25793
    raw/Frag20_14_sdf_QM.pt:   30331
    raw/Frag20_15_sdf_QM.pt:   31719
    raw/Frag20_16_sdf_QM.pt:   35584
    raw/Frag20_17_sdf_QM.pt:   36201
    raw/Frag20_18_sdf_QM.pt:   32501
    raw/Frag20_19_sdf_QM.pt:   23474
    :return:
    """
    files = glob.glob('raw/Frag20*_QM.npz')
    for file in files:
        data = np.load(file)
        print("{}:   {}".format(file, len(data)))

    return


def view_frag20_data_combined():
    """
    test: 56636
    train: 509660
    :return:
    """
    data_train = np.load('raw/fragment20_all_QM_train.npz')
    data_test = np.load('raw/fragment20_all_QM_test.npz')
    split = np.load('raw/split.npz')
    return


def view_frag20_split():
    """

    :return:
    """
    data = {}
    for i in range(9, 21):
        data[i] = np.load('raw/fragment{}_split.npz'.format(i))
    split = np.load('raw/split.npz')
    return


def view_qm9_data():
    """

    :return:
    """
    data1 = np.load('raw/qm9_qm_removeproblem.npz')
    data2 = torch.load('raw/QM9SDFs_removeproblem.pt')
    return


def gen_qm9_transformer_input():
    target = np.load('raw/qm9_qm_removeproblem.npz')
    sdf = torch.load('raw/QM9SDFs_removeproblem.pt')
    split = np.load('processed/split_qm9.npz')
    for tag in ['train', 'validation', 'test']:
        index = split[tag]
        with open('qm9-{}.csv'.format(tag), 'w') as f:
            f.write('id,SRC,TRG\n')
            for i in index:
                f.write('{},'.format(i))
                f.write('{},'.format(Chem.MolToSmiles(sdf[i])))
                f.write("{},".format(target['E'][i]))
                f.write('\n')
    valid_data = pd.read_csv('qm9-validation')
    return


def gen_frag20_transformer_input():
    with open('frag20-all.csv', 'w') as f:
        f.write('ID,SMILES,Energy(eV)\n')
        mol_id = 0
        for n_heavy in tqdm(range(9, 21)):
            _name = 'nolarger9' if n_heavy == 9 else n_heavy
            target = np.load('raw/frag20/Frag20_{}_PhysNet_QM.npz'.format(_name))['E']
            sdf = torch.load('raw/frag20/Frag20_{}_sdf_QM.pt'.format(_name))
            for mol, energy in zip(sdf, target):
                f.write('{},{},{}\n'.format(mol_id, Chem.MolToSmiles(Chem.RemoveHs(mol), allHsExplicit=False), energy))
                mol_id += 1
    return


def split_frag20_smiles():
    data = pd.read_csv("frag20-all.csv")
    split_tensor = torch.randperm(len(data))
    train_index = split_tensor[:-41000]
    val_index = split_tensor[-41000:-40000]
    test_index = split_tensor[-40000:]
    for index, name in zip([train_index, val_index, test_index], ["train", "validation", "test"]):
        data.iloc[index].to_csv("frag20-{}.csv".format(name), index=False, header=False)


def test_torch_scatter():
    x = torch.rand(32, 7)
    s = scatter(dim=-1, index=torch.as_tensor([0, 0, 0, 1, 1, 2, 2]), src=x)
    return


def view_conf20_data():
    physnet = np.load('raw/conf20_QM_PhysNet.npz', allow_pickle=True)
    sdf = torch.load('raw/conf20_QM.pt')
    conf20_smiles = set([Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in sdf])
    frag20_sdf = torch.load('raw/Frag20_20_sdf_QM.pt')
    frag20_smiles = set([Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in frag20_sdf])

    intersect = conf20_smiles.intersection(frag20_smiles)
    largest_mol = 0
    for mol in sdf:
        num = mol.GetNumAtoms(onlyExplicit=False)
        if num > largest_mol:
            largest_mol = num
    return


def view_e_mol9():
    mol_list = torch.load("raw/eMol9_QM.pt")
    phys_input = np.load("raw/eMol9_PhysNet_QM.npz", allow_pickle=True)
    return


if __name__ == '__main__':
    split_frag20_smiles()
