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
            train_index = split_data["train_index"]
            perm_matrix = torch.randperm(len(train_index))
            self.train_index = train_index[perm_matrix[:-1000]]
            self.val_index = train_index[perm_matrix[-1000:]]
            self.test_index = split_data["test_index"]

    @property
    def raw_file_names(self):
        return ["dummy"]

    @property
    def processed_file_names(self):
        return [self.dataset_name, self.split]

    def download(self):
        pass

    def process(self):
        pass


