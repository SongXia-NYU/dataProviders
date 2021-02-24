import pandas as pd
from tqdm import tqdm

dd_csv = pd.read_csv("frag20_solvation.csv")


def get_col_name(i):
    if i == 9:
        return "opt_Inchi_h"
    else:
        return "QM_InChI"


if __name__ == '__main__':
    frag_inchi = {i: pd.read_csv("Frag20_{}_1D_infor.csv".format(i))[get_col_name(i)] for i in range(9, 21)}
    loss_data = pd.DataFrame()

    for i in tqdm(range(len(dd_csv))):
        InChI = dd_csv["InChI"][i]
        source = dd_csv["SourceFile"][i]
        if source == "ccdc":
            continue
        elif source == "less10":
            source = 9
        else:
            source = int(source)
        if not (InChI in frag_inchi[source]):
            loss_data = loss_data.append(dd_csv.iloc[i])

    loss_data.to_csv("lost_csv.csv")


