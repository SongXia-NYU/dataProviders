import rdkit
import torch_geometric
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import MMFFOptimizeMolecule, MMFFOptimizeMoleculeConfs
from rdkit.Chem.rdmolfiles import SDWriter
from DataGen.genconfs import runGenerator, gen_conformers
import pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np
import os.path as osp
import torch
import argparse
import multiprocess
from multiprocess.pool import Pool

from DataPrepareUtils import my_pre_transform
from GaussUtils.GaussInfo import Gauss16Info


def free_solv_sdfs():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--start", type=int)
    # parser.add_argument("--end", type=int)
    # args, _ = parser.parse_known_args()
    dd_csv_folder = "/scratch/projects/yzlab/group/temp_dd/solvation/calculated/"
    train_csv = pd.read_csv(osp.join(dd_csv_folder, "train.csv"))
    valid_csv = pd.read_csv(osp.join(dd_csv_folder, "valid.csv"))
    test_csv = pd.read_csv(osp.join(dd_csv_folder, "test.csv"))
    # concatenate them in this order
    error_list = torch.load("conf_error_list.pt")
    concat_csv = pd.concat([train_csv, valid_csv, test_csv], ignore_index=True).iloc[error_list]
    runGenerator(concat_csv.index.tolist(), concat_csv["SMILES"].tolist(), "solvation",
                 "/scratch/sx801/data/sol-frag20-ccdc/mmff_confs")


def mmff_min_sdfs():
    file_pattern = "/scratch/sx801/data/sol-frag20-ccdc/mmff_confs/{}_confors.sdf"
    dd_csv_folder = "/scratch/projects/yzlab/group/temp_dd/solvation/calculated/"
    train_csv = pd.read_csv(osp.join(dd_csv_folder, "train.csv"))
    valid_csv = pd.read_csv(osp.join(dd_csv_folder, "valid.csv"))
    test_csv = pd.read_csv(osp.join(dd_csv_folder, "test.csv"))
    # concatenate them in this order
    concat_csv = pd.concat([train_csv, valid_csv, test_csv], ignore_index=True)
    dst_folder = "/ext3/mmff_sdfs/"
    error_list = []

    def convert_conf_to_sdf_i(i):
        try:
            f = file_pattern.format(i)
            lowest_e = np.inf
            selected_mol = None

            suppl = rdkit.Chem.SDMolSupplier(f, removeHs=False)
            for mol in suppl:
                prop_dict = mol.GetPropsAsDict()
                if lowest_e > prop_dict["energy_abs"]:
                    selected_mol = mol
            w = SDWriter(dst_folder + "/{}.mmff.sdf".format(osp.basename(f).split("_")[0]))
            w.write(selected_mol)
            print("success: {}".format(i))
        except Exception:
            # too broad exception but who cares?
            print("error: {}".format(i))
            error_list.append(i)

    with Pool(10) as p:
        p.map(convert_conf_to_sdf_i, concat_csv.index.tolist())

    torch.save(error_list, "conf_error_list.pt")


def convert_pt():
    sdf_folder = "raw/freesolv_sdfs"
    target = pd.read_csv("raw/freesolv.csv")
    dst = "processed/freesolv_mmff.pt"
    split_dst = "processed/freesolv_split.pt"
    data_list = []
    for i in tqdm(target.index):
        sdf_f = osp.join(sdf_folder, "{}.mmff.sdf".format(i))
        more_target = {"activity": torch.as_tensor(target["activity"][i]).view(-1),
                       "group": np.array([target["group"][i]], dtype=object).reshape(-1)}
        info = Gauss16Info(qm_sdf=sdf_f, prop_dict_raw={"dd_target": more_target})
        data_edge = my_pre_transform(info.get_torch_data(), edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                     cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                     bond_atom_sep=False, record_long_range=True)
        data_list.append(data_edge)

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               osp.join(dst))
    group = target["group"].values
    split = {"train_index": target.index[group == "train"],
             "valid_index": target.index[group == "valid"],
             "test_index": target.index[group == "test"]}
    torch.save(split, split_dst)


if __name__ == '__main__':
    free_solv_sdfs()
    print("finished")
