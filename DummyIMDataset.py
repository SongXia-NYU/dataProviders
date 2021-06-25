import torch
from torch_geometric.data import InMemoryDataset


class DummyIMDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, split=None, **kwargs):
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_index, self.val_index, self.test_index = None, None, None
        if split is not None:
            split_data = torch.load(self.processed_paths[1])
            self.test_index = split_data["test_index"]
            if ("valid_index" not in split_data) and ("val_index" not in split_data):
                train_index = split_data["train_index"]
                perm_matrix = torch.randperm(len(train_index))
                self.train_index = train_index[perm_matrix[:-1000]]
                self.val_index = train_index[perm_matrix[-1000:]]
            else:
                self.train_index = split_data["train_index"]
                for name in ["val_index", "valid_index"]:
                    if name in split_data.keys():
                        self.val_index = split_data[name]

    @property
    def raw_file_names(self):
        return ["dummy"]

    @property
    def processed_file_names(self):
        return [self.dataset_name, self.split] if self.split is not None else [self.dataset_name]

    def download(self):
        pass

    def process(self):
        pass
>>>>>>> 7b3729cb67ccf9ce30de08ab8a00fce0cc13c559
