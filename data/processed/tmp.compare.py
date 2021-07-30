import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import r2_score
import os.path as osp


def get_data(pt_path, csv_path, root="."):
    data_pt = torch.load(osp.join(root, pt_path))
    data_csv = pd.read_csv(osp.join(root, csv_path))
    assert len(data_csv) == len(data_pt["activity"])
    print(f"difference: {(torch.as_tensor(data_csv['activity'])-data_pt['activity']).abs().sum()}")

    data_ready = {}
    for split in ["train", "valid", "test"]:
        mask = (data_csv["group"].values == split)
        data_ready[f"{split}_X"] = data_pt["mol_prop"][mask]
        data_ready[f"{split}_y"] = data_pt["activity"][mask]

    return data_ready


if __name__ == '__main__':
    data = get_data("free_solv_mol_prop.pt", "freesolv.csv")
    for split in ["train", "valid", "test"]:
        print(f"----split {split}-----")
        pred_sol = (data[f"{split}_X"][:, 1] - data[f"{split}_X"][:, 0]) * 23.0061
        exp_sol = data[f"{split}_y"]
        mse = nn.MSELoss()
        print(f"RMSE: {torch.sqrt(mse(pred_sol, exp_sol))}")
        print(f"R2 score {r2_score(exp_sol, pred_sol)}")
        print("--------")
    print("finished")
