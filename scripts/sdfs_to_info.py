import pandas as pd
import rdkit
from rdkit.Chem import MolFromSmiles, SDMolSupplier
from glob import glob
import os.path as osp


def sdf_to_info(sdf_folder, save_folder):
    sdf_files = glob(osp.join(sdf_folder, "*.sdf"))
    result = pd.DataFrame()
    for sdf in sdf_files:
        f_id = osp.basename(sdf).split(".")[0]
        this_info = {"file_name": f_id}
        mol = list(SDMolSupplier(sdf))[0]
        this_info["SMILES"] = mol.GetProp("SMILES")
        this_info["n_heavy"] = mol.GetNumHeavyAtoms()
        result = result.append(this_info, ignore_index=True)
    result = result.sort_values(by="n_heavy")
    result.to_csv(osp.join(save_folder, "info.csv"), index=False)


if __name__ == '__main__':
    sdf_to_info("../sol_data/freesolv_sol/freesolv_sdfs", "../sol_data/freesolv_sol")
