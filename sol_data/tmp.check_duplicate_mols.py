import pandas as pd
from tqdm import tqdm
import os.path as osp
from rdkit.Chem import MolFromSmiles, MolToInchi
import torch

if __name__ == '__main__':
    dd_csv_folder = "/scratch/projects/yzlab/group/temp_dd/solvation/calculated/"
    train_csv = pd.read_csv(osp.join(dd_csv_folder, "train.csv"))
    valid_csv = pd.read_csv(osp.join(dd_csv_folder, "valid.csv"))
    test_csv = pd.read_csv(osp.join(dd_csv_folder, "test.csv"))

    lipop_csv = pd.read_csv("lipop.csv")
    lipop_inchi = [MolToInchi(MolFromSmiles(s)) for s in lipop_csv["cano_smiles"]]
    freesolv_csv = pd.read_csv("freesolv.csv")
    freesolv_inchi = [MolToInchi(MolFromSmiles(s)) for s in freesolv_csv["cano_smiles"]]

    all_inchi = lipop_inchi
    all_inchi.extend(freesolv_inchi)

    inchi_exist_map = []

    # concatenate them in this order
    concat_csv = pd.concat([train_csv, valid_csv, test_csv], ignore_index=True)
    for inchi in tqdm(concat_csv["InChI"]):
        if inchi in all_inchi:
            inchi_exist_map.append(1)
        else:
            inchi_exist_map.append(0)

    inchi_exist_map = torch.as_tensor(inchi_exist_map).long()
    print("{} / {}".format(inchi_exist_map.sum(), inchi_exist_map.shape[0]))
    torch.save(inchi_exist_map, "inchi_exist_map.pt")
