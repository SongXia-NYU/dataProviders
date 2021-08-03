from mendeleev import get_all_elements
import torch

dataset_name = "freesolv_mmff.pt"

if __name__ == '__main__':
    atm_num_to_name = {e.atomic_number: e.symbol for e in get_all_elements()}
    dataset = torch.load(dataset_name)
    Z = dataset[0].Z
    atom_numbers = set(Z.tolist())
    with open(f"{dataset_name.split('.')[0]}_atoms.txt", "w") as f:
        f.write("atomic_number symbol\n")
        for num in atom_numbers:
            f.write(f"{num},{atm_num_to_name[num]}\n")

