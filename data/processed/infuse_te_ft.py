import torch
from copy import copy

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15


def infuse_te_ft_free_solv():
    calc_mol_prop = torch.load("free_solv_embed160_exp331_dataset.pt")
    data = torch.load("freesolv_mmff.pt")
    for i, key in enumerate(["gasEnergy", "watEnergy", "octEnergy"]):
        setattr(data[0], key, calc_mol_prop["mol_prop"][:, i])
        data[1][key] = copy(data[1]["N"])
    setattr(data[0], "CalcOct", (data[0].octEnergy - data[0].gasEnergy)*23.06035)
    setattr(data[0], "CalcSol", calc_mol_prop["activity"])
    setattr(data[0], "watEnergy", calc_mol_prop["activity"]/23.06035 + data[0].gasEnergy)
    setattr(data[0], "watOct", (data[0].watEnergy - data[0].octEnergy) * 23.06035)

    for key in ["CalcOct", "CalcSol", "watOct"]:
        data[1][key] = copy(data[1]["N"])

    torch.save(data, "freesolv_te_mam_ft_mmff_exp331.pt")

    print("finished")


def infuse_te_ft_lipop():
    calc_mol_prop = torch.load("lipop_embed160_exp331_dataset.pt")
    data = torch.load("lipop_logP_mmff.pt")
    for i, key in enumerate(["gasEnergy", "watEnergy", "octEnergy"]):
        setattr(data[0], key, calc_mol_prop["mol_prop"][:, i])
        data[1][key] = copy(data[1]["N"])

    pred_watOct = (data[0].watEnergy - data[0].octEnergy) * 23.06035

    setattr(data[0], "watOct", calc_mol_prop["activity"] * logP_to_watOct)
    setattr(data[0], "watEnergy", data[0].watOct/23.06035 + data[0].octEnergy)
    setattr(data[0], "CalcOct", (data[0].octEnergy - data[0].gasEnergy)*23.06035)
    setattr(data[0], "CalcSol", (data[0].watEnergy - data[0].gasEnergy)*23.06035)

    mae = (pred_watOct - data[0].watOct).abs().mean()
    print(f"MAE between pred and exp: {mae}")

    for key in ["CalcOct", "CalcSol", "watOct"]:
        data[1][key] = copy(data[1]["N"])

    torch.save(data, "lipop_te_mam_ft_mmff_exp331.pt")

    print("finished")


if __name__ == '__main__':
    # infuse_te_ft_lipop()
    print(logP_to_watOct)
