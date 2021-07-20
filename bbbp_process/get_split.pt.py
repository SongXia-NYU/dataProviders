import pandas as pd
import torch
import rdkit

from rdkit.Chem.AllChem import MolFromSmiles, MolToInchiKey


if __name__ == '__main__':
    paper_csv = pd.read_csv("bbbp_graph_paper.csv")
    cano_smiles = paper_csv["cano_smiles"].tolist()
    inchi_list = [MolToInchiKey(MolFromSmiles(smiles)) for smiles in cano_smiles]

    smiles_processed = torch.load("bbbp_mmff.pt")[0].smiles

    split = {
        "train_index": [],
        "valid_index": [],
        "test_index": []
    }

    for i, smiles in enumerate(smiles_processed):
        try:
            inchi = MolToInchiKey(MolFromSmiles(smiles))
        except Exception as e:
            print(e)
            continue
        try:
            num = inchi_list.index(inchi)
            group = paper_csv["group"][num]
            split[f"{group}_index"].append(i)
        except ValueError as e:
            print(e)

    for key in split.keys():
        split[key] = torch.as_tensor(split[key]).long()
    torch.save(split, "split_bbbp_mmff.pt")
    print("hello")
