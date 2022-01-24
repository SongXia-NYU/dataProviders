import torch
import torch_geometric
import os.path as osp

from DummyIMDataset import DummyIMDataset


def lite_gen():
    dataset_name = ""
    save_name = "acsf308_lite.pt"
    save_split_name = "acsf308_lite_split.pt"

    dataset = DummyIMDataset(root="../data", dataset_name=dataset_name)

    data_list = []
    for i in range(10000):
        data_list.append(dataset[i])

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), osp.join("../data/processed", save_name))

    split = {
        "train_index": torch.arange(9000),
        "val_index": torch.arange(9000, 9100),
        "test_index": torch.arange(9100, 10000)
    }
    torch.save(split, save_split_name)


if __name__ == '__main__':
    lite_gen()
