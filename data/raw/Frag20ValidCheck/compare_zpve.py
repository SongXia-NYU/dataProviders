import pandas as pd
import numpy as np
from ase.units import Hartree, eV
import rdkit
from math import sqrt
from rdkit.Chem import MolFromInchi
from rdkit.Chem.Draw import MolsToGridImage

if __name__ == '__main__':
    dd_csv = pd.read_csv("frag20_solvation.csv")
    dd_csv_dict = {i: dd_csv.loc[dd_csv["SourceFile"].values.astype(np.str) == str(i)] for i in range(15, 21)}
    jl_info_csv_dict = {i: pd.read_csv("Frag20_{}_1D_infor.csv".format(i)) for i in range(15, 21)}
    jl_target_csv_dict = {i: pd.read_csv("Frag20_{}_target.csv".format(i)) for i in range(15, 21)}
    calc_csv100 = pd.read_csv("frag20-100.csv")
    result_csv = pd.DataFrame()
    for i in range(len(calc_csv100)):
        tmp_dict = {"f_name": calc_csv100["f_name"][i].split(".")[0]}
        n_heavy = int(tmp_dict["f_name"].split("_")[0])
        dd_id = int(tmp_dict["f_name"].split("_")[1])
        this_dd_data = dd_csv_dict[n_heavy].loc[dd_csv_dict[n_heavy]["ID"].values.astype(np.int) == dd_id]
        tmp_dict["InChI"] = this_dd_data["InChI"].item()
        tmp_dict["SMILES"] = this_dd_data["SMILES"].item()
        this_jl_info_data = jl_info_csv_dict[n_heavy].loc[jl_info_csv_dict[n_heavy]["QM_InChI"] == tmp_dict["InChI"]]
        this_jl_target_data = jl_target_csv_dict[n_heavy].loc[jl_target_csv_dict[n_heavy]["index"] == this_jl_info_data["index"].item()]
        tmp_dict["jl_gas_E(eV)"] = this_jl_target_data["U0"].item()
        tmp_dict["sx_gas_E(eV)"] = calc_csv100["U0"][i]
        tmp_dict["dd_gas_E(eV)"] = this_dd_data["gasEnergy"].item() * (Hartree / eV)
        tmp_dict["difference(jl-dd)(eV)"] = tmp_dict["jl_gas_E(eV)"] - tmp_dict["dd_gas_E(eV)"]
        tmp_dict["jl_zpve(eV)"] = this_jl_target_data["zpve"].item()
        tmp_dict["sx_zpve(eV)"] = calc_csv100["zpve"][i]
        result_csv = result_csv.append(pd.DataFrame(tmp_dict, index=[i]))

    result_csv.to_csv("frag20-100-compare.csv")
    mols = [MolFromInchi(item) for item in result_csv["InChI"].values]
    MolsToGridImage(mols, int(sqrt(len(mols))), subImgSize=(400, 400)).save("mols.png")

