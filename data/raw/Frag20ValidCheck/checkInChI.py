import pandas as pd
import torch
from rdkit.Chem import MolToInchi
from Frag20ValidCheck.generate_loss_data import get_col_name


diff_data = pd.DataFrame()

for n_heavy in range(9, 21):
    mols = torch.load("Frag20_{}_QM.pt".format(n_heavy))
    info_csv = pd.read_csv("Frag20_{}_1D_infor.csv".format(n_heavy))
    assert len(mols) == len(info_csv)
    for i in range(len(mols)):
        mol_inchi = MolToInchi(mols[i])
        csv_inchi = info_csv[get_col_name(n_heavy)][i]
        if mol_inchi != csv_inchi:
            diff_data = diff_data.append(pd.DataFrame(data={
                "n_heavy": [n_heavy],
                "index": [i],
                "mol_inchi": [mol_inchi],
                "csv_inchi": [csv_inchi]
            }))

diff_data.to_csv("diff_data.csv")
