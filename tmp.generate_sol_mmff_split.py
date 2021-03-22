import pandas as pd
import os.path as osp
import torch


if __name__ == '__main__':
    dd_csv_folder = "/scratch/projects/yzlab/group/temp_dd/solvation/calculated/"
    train_csv = pd.read_csv(osp.join(dd_csv_folder, "train.csv"))
    valid_csv = pd.read_csv(osp.join(dd_csv_folder, "valid.csv"))
    test_csv = pd.read_csv(osp.join(dd_csv_folder, "test.csv"))

    train_index = []
    valid_index = []
    test_index = []

    success_map = torch.load("success_map.pt")
    train_map = torch.zeros_like(success_map)
    valid_map = torch.zeros_like(success_map)
    test_map = torch.zeros_like(success_map)
    train_map[:train_csv.shape[0]] = 1
    valid_map[train_csv.shape[0]:train_csv.shape[0]+valid_csv.shape[0]] = 1
    test_map[-test_csv.shape[0]:] = 1

    train_mask = train_map & success_map
    valid_mask = valid_map & success_map
    test_mask = test_map & success_map

    save_root = "data/processed"
    torch.save({"train_index": train_mask.nonzero().reshape(-1),
                "valid_index": valid_mask.nonzero().reshape(-1),
                "test_index": test_mask.nonzero().reshape(-1)},
               osp.join(save_root, "frag20_sol_split_mmff_gen_03222021.pt"))
