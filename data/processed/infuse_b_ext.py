import torch
import torch_geometric.data

from DataPrepareUtils import extend_bond
from DummyIMDataset import DummyIMDataset
import os.path as osp


if __name__ == '__main__':
    dataset_name = "frag20reducedAllSolRef-Bmsg-cutoff-10.00-sorted-defined_edge-lr-MMFF.pt"
    split = "frag20_sol_split.pt"
    save_name = "frag20reducedAllSolRef-B-ext-cutoff-10.00-sorted-defined_edge-lr-MMFF.pt"
    root = "/scratch/sx801/scripts/physnet-dimenet/dataProviders/data"

    dataset = DummyIMDataset(root=root, dataset_name=dataset_name, split=split)
    data_list = []
    for i in range(len(dataset)):
        data = dataset[i]
        edge = extend_bond(data.B_edge_index)
        setattr(data, "B-ext_edge_index", edge)
        setattr(data, "num_B-ext_edge", edge.shape[-1])
        data_list.append(data)
        if i % 1000 == 0:
            print(f"---------{i} / {len(dataset)}---------")

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), osp.join(root, "processed", save_name))
