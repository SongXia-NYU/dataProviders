import os
import os.path as osp
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    n_heavy = 16
    DD2JL_csv_f = osp.join("/ext3/Frag20_{}/Frag20_{}_frag_index.csv".format(n_heavy, n_heavy))
    DD2JL_csv = pd.read_csv(DD2JL_csv_f)
    failed_csv = pd.read_csv("octanol_fail_{}.csv".format(n_heavy))
    failed_JL_index = [DD2JL_csv.loc[DD2JL_csv["idx_name"] == dd_index]["frags_index"].item()
                       for dd_index in failed_csv.values.reshape(-1).tolist()]

    print(failed_JL_index)
    exit(-1)

    dst = osp.join("/ext3/Frag20_failed_mols/{}/".format(n_heavy))
    os.makedirs(dst, exist_ok=True)

    for i in tqdm(failed_JL_index):
        src = osp.join("/ext3/Frag20_{}_data/{}.opt.sdf".format(n_heavy, i))
        os.system("cp {} {}".format(src, dst))
