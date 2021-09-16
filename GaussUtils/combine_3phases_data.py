import pandas as pd
import torch
import torch_geometric.data
from ase.units import Hartree, eV
import os
import os.path as osp

from DataPrepareUtils import my_pre_transform
from DummyIMDataset import DummyIMDataset

hartree2ev = Hartree / eV
kcal2ev = 1 / 23.06035

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15


def _diff_energy_kcal_mol(col1, col2, df: pd.DataFrame):
    # TODO: test it
    e1 = df[col1]
    e2 = df[col2]
    diff = e1 - e2
    diff_unit = diff / kcal2ev
    return diff_unit


def combine_csv(gas_csv, water_csv, oct_csv, paper_csv, out_csv):
    gas_df = pd.read_csv(gas_csv).set_index("f_name").rename(columns={"E": "gasEnergy"})
    water_df = pd.read_csv(water_csv).set_index("f_name").rename(columns={"E": "watEnergy"})
    oct_df = pd.read_csv(oct_csv).set_index("f_name").rename(columns={"E": "octEnergy"})
    result = gas_df.join(water_df, lsuffix="", rsuffix="_water")
    result = result.join(oct_df, rsuffix="_oct")

    result["CalcSol"] = _diff_energy_kcal_mol("watEnergy", "gasEnergy", result)
    result["CalcOct"] = _diff_energy_kcal_mol("octEnergy", "gasEnergy", result)
    result["watOct"] = _diff_energy_kcal_mol("watEnergy", "octEnergy", result)
    result["CalcLogP"] = result["watOct"] / logP_to_watOct

    paper_df = pd.read_csv(paper_csv)
    result["group"] = [paper_df.iloc[int(i)]["group"] for i in result.index]
    result["cano_smiles"] = [paper_df.iloc[int(i)]["cano_smiles"] for i in result.index]
    result["activity"] = [paper_df.iloc[int(i)]["activity"] for i in result.index]

    result.to_csv(out_csv)


def infuse_energy(sol_csv, dataset_p, dataset_root):
    dataset = DummyIMDataset(root=dataset_root, dataset_name=dataset_p)
    sol_df = pd.read_csv(sol_csv).set_index("f_name")
    data_list = []
    current_idx = 0
    split = {"train_index": [],
             "valid_index": [],
             "test_index": []}
    for i in range(len(dataset)):
        this_data = dataset[i]
        this_idx = int(this_data["f_name"])
        if this_idx in sol_df.index:
            info = sol_df.loc[this_idx]
            for key in ["gasEnergy", "watEnergy", "octEnergy", "CalcSol", "CalcOct", "watOct", "activity", "CalcLogP"]:
                setattr(this_data, key, torch.as_tensor(info[key]))

            group = info["group"]
            split[f"{group}_index"].append(current_idx)
            current_idx += 1

            this_data = my_pre_transform(this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                         cutoff=10.0, boundary_factor=100., use_center=True, mol=None,
                                         cal_3body_term=False,
                                         bond_atom_sep=False, record_long_range=True)
            data_list.append(this_data)

    collated = torch_geometric.data.InMemoryDataset.collate(data_list)
    torch.save(collated, osp.join(dataset_root, "processed", dataset_p.split(".")[0] + ".pt"))
    torch.save(split, osp.join(dataset_root, "processed", "split_" + dataset_p.split(".")[0] + ".pt"))
