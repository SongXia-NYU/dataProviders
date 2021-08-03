import torch
from copy import copy


def infuse_te_ft():
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


if __name__ == '__main__':
    infuse_te_ft()
