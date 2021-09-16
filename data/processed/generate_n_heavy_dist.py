import torch
from DummyIMDataset import DummyIMDataset
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataset = DummyIMDataset(root="..", dataset_name="frag20jianing-Bmsg-cutoff-10.00-sorted-defined_edge-lr-MMFF(sample-uniform)-test.pt", split=None)
    n_heavy_list = []
    activity_list = []
    for i in range(len(dataset)):
        n_heavy = (dataset[i].Z > 1).sum().item()
        n_heavy_list.append(n_heavy)
        # activity_list.append(dataset[i].activity)

    plt.hist(n_heavy_list, bins=range(min(n_heavy_list), max(n_heavy_list)+1))
    plt.xlabel("num of heavy atoms")
    plt.ylabel("count")
    plt.title("Frag20 test set, n heavy distribution")
    plt.savefig("frag20_test")
    plt.show()
    print("je")

