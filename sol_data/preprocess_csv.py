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

from DataPrepareUtils import my_pre_transform
from GaussUtils.GaussInfo import Gauss16Info


def free_solv_sdfs():
    data = pd.read_csv("raw/freesolv.csv")
    runGenerator(data.index.tolist(), data["cano_smiles"].tolist(), "freesolv", "raw/freesolv_confs")


def mmff_min_sdfs():
    files = glob("raw/freesolv_confs/*_confors.sdf")
    dst_folder = "raw/freesolv_sdfs"
    for f in tqdm(files):
        lowest_e = np.inf
        selected_mol = None

        suppl = rdkit.Chem.SDMolSupplier(f, removeHs=False)
        for mol in suppl:
            prop_dict = mol.GetPropsAsDict()
            if lowest_e > prop_dict["energy_abs"]:
                selected_mol = mol
        w = SDWriter(dst_folder + "/{}.mmff.sdf".format(osp.basename(f).split("_")[0]))
        w.write(selected_mol)


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
    convert_pt()
    print("finished")
