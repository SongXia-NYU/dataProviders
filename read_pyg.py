import torch

from DummyIMDataset import DummyIMDataset


def read_pyg():
    dataset = DummyIMDataset(root="data", dataset_name="frag20-ultimate-sol-qm-04182022.pyg",
                             split="frag20-ultimate-sol-split-qm-04182022.pyg")
    print(f"Size: {len(dataset)}")
    training_data = dataset[torch.as_tensor(dataset.train_index)]
    print(f"Training size: {len(training_data)}")
    print("Finished")


if __name__ == '__main__':
    read_pyg()
