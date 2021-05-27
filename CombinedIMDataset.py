import torch
from torch_geometric.data import InMemoryDataset


class CombinedIMDataset(InMemoryDataset):
    @property
    def raw_file_names(self):
        return ["dummy"]

    @property
    def processed_file_names(self):
        return [self.dataset_name]

    def download(self):
        pass

    def process(self):
        assert self.dataset_list is not None
        data_list = []
        for dataset in self.dataset_list:
            data_list.extend([dataset[i] for i in range(len(dataset))])
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __init__(self, root, dataset_name, dataset_list=None, transform=None, pre_transform=None, train_index=None,
                 val_index=None, test_index=None, **kwargs):
        self.dataset_list = dataset_list
        self.dataset_name = dataset_name
        super().__init__(root, transform, pre_transform)
        del self.dataset_list
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_index, self.val_index, self.test_index = train_index, val_index, test_index


if __name__ == '__main__':
    pass
