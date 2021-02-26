import torch
from torch_geometric.data import InMemoryDataset


class DummyIMDataset(InMemoryDataset):
    def __init__(self, root, dataset_name):
        self.dataset_name = dataset_name
        super().__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_index, self.val_index, self.test_index = None, None, None

    @property
    def raw_file_names(self):
        return ["dummy"]

    @property
    def processed_file_names(self):
        return [self.dataset_name]

    def download(self):
        pass

    def process(self):
        pass


