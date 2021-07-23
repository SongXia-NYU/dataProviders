from multiprocessing import Pool

from DataGen.genconfs import runGenerator
import pandas as pd
import os.path as osp
import torch


def _run_generator(i):
    print(""+concat_csv["SMILES"].tolist()[i])
    runGenerator([error_list[i]], [concat_csv["SMILES"].tolist()[i]],
                 "sol", "/scratch/sx801/data/sol-frag20-ccdc/mmff_confs/")


if __name__ == '__main__':
    dd_csv_folder = "/scratch/projects/yzlab/group/temp_dd/solvation/calculated/"
    train_csv = pd.read_csv(osp.join(dd_csv_folder, "all.csv"))
    valid_csv = pd.read_csv(osp.join(dd_csv_folder, "valid.csv"))
    test_csv = pd.read_csv(osp.join(dd_csv_folder, "test.csv"))
    # concatenate them in this order
    # error_list = torch.load("conf_error_list.pt")
    error_list = [1668]
    concat_csv = pd.concat([train_csv, valid_csv, test_csv], ignore_index=True).iloc[error_list]
    with Pool(20) as p:
        p.map(_run_generator, range(len(error_list)))
