import torch
from DummyIMDataset import DummyIMDataset
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataset = DummyIMDataset(root="..", dataset_name="freesolv_mmff.pt", split="freesolv_split.pt")
    n_heavy_list = []
    activity_list = []
    for i in range(len(dataset)):
        n_heavy = (dataset[i].Z > 1).sum().item()
        n_heavy_list.append(n_heavy)
        activity_list.append(dataset[i].activity)

    plt.hist(n_heavy_list, bins=range(min(n_heavy_list), max(n_heavy_list)+1))
    plt.xlabel("num of heavy atoms")
    plt.ylabel("count")
    plt.savefig("freeSolv_dist")
    plt.show()
    print("je")

