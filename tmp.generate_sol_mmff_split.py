import pandas as pd
import os.path as osp
import torch

if __name__ == '__main__':
    is_generated_mmff = False
    dd_csv_folder = "/scratch/projects/yzlab/group/temp_dd/solvation/calculated/"
    train_csv = pd.read_csv(osp.join(dd_csv_folder, "train.csv"))
    valid_csv = pd.read_csv(osp.join(dd_csv_folder, "valid.csv"))
    test_csv = pd.read_csv(osp.join(dd_csv_folder, "test.csv"))

    if is_generated_mmff:
        success_map = torch.load("success_map.pt")
        train_size = success_map[:train_csv.shape[0]].sum()
        valid_size = success_map[train_csv.shape[0]:train_csv.shape[0] + valid_csv.shape[0]].sum()
        test_size = success_map[-test_csv.shape[0]:].sum()
    else:
        train_size = train_csv.shape[0]
        valid_size = valid_csv.shape[0]
        test_size = test_csv.shape[0]
        success_map = torch.ones(train_size+valid_size+test_size)

    duplicate_map = torch.load("sol_data/inchi_exist_map.pt")
    dup_to_suc_map = duplicate_map[torch.nonzero(success_map).view(-1)]

    save_root = "data/processed"
    torch.save({"train_index": torch.arange(train_size)[dup_to_suc_map[:train_size] == 0],
                "valid_index":
                    torch.arange(train_size, train_size + valid_size)[
                        dup_to_suc_map[train_size: train_size + valid_size] == 0],
                "test_index":
                    torch.arange(train_size + valid_size, train_size + valid_size + test_size)[
                        dup_to_suc_map[-test_size:] == 0]},
               osp.join(save_root, "frag20_sol_split_{}03222021.pt".format("mmff_gen_" if is_generated_mmff else "")))
