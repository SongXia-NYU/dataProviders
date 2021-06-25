"""
Batch move files according to its id recorded in a *.csv file
"""
import argparse
import os
import os.path as osp
import pandas as pd
from glob import glob
from tqdm import tqdm


def _move(n_heavy, csv_pattern, src_dir, dst_dir, src_pattern):
    src_dir = src_dir.format(n_heavy)
    dst_dir = osp.join(dst_dir, "{}".format(n_heavy))
    os.makedirs(dst_dir, exist_ok=True)

    failed_list = pd.read_csv(csv_pattern.format(n_heavy))
    in_files = [osp.join(src_dir, src_pattern.format(i)) for i in failed_list.values.reshape(-1).tolist()]

    for f in tqdm(in_files):
        os.system("cp {} {} ".format(f, dst_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_heavy", type=int)
    parser.add_argument("--csv_pattern", type=str, default="octanol_fail_{}.csv")
    parser.add_argument("--src_dir", type=str, default="/ext3/Frag20_{}/octanol_smd_out/")
    parser.add_argument("--dst_dir", type=str, default="/ext3/Frag20_failed_mols/")
    parser.add_argument("--src_pattern", type=str, default="{}_gas_opt_octanol_smd.log")
    args = parser.parse_args()
    args_dict = vars(args)

    _move(**args_dict)


if __name__ == '__main__':
    main()
