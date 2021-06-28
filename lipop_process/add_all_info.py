import torch
import pandas as pd
import torch_geometric
from tqdm import tqdm

from DataPrepareUtils import my_pre_transform
from DummyIMDataset import DummyIMDataset


def main():
    data = DummyIMDataset(".", "lipop_target.pt")
    target_csv = pd.read_csv("lipop_target.csv")
    assert len(data) == len(target_csv)

    source_csv = pd.read_csv("../sol_data/lipop.csv")

    split = {
        "train_index": [],
        "valid_index": [],
        "test_index": []
    }
    data_list = []
    for i in tqdm(range(len(data))):
        this_data = data[i]
        source = int(target_csv["f_name"][i])
        this_data_edge = my_pre_transform(this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                          cutoff=10.0, boundary_factor=100., use_center=True, mol=None,
                                          cal_3body_term=False, bond_atom_sep=False, record_long_range=True)
        this_data_edge.source = torch.as_tensor(int(source)).long()
        expLogP = source_csv["activity"][source]
        this_data_edge.expLogP = torch.as_tensor(expLogP).double()

        group = source_csv["group"][source]
        split[f"{group}_index"].append(i)

        data_list.append(this_data_edge)
    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), "lipop_logP.pt")
    for key in split.keys():
        split[key] = torch.as_tensor(split[key]).long()
    torch.save(split, "split_lipop_logP.pt")


if __name__ == '__main__':
    main()
