"""
Copy the failed Gaussian calculation's *.com file into a new folder.
For a second round of calculation
"""
import shutil
from glob import glob
import os
import os.path as osp
from GaussUtils.GaussInfo import Gauss16Info

from tqdm import tqdm


def move_failed_mol_coms(log_dir, com_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for log_file in tqdm(glob(osp.join(log_dir, "*.log"))):
        info = Gauss16Info(log_path=log_file)
        if not info.normal_termination:
            mol_id = osp.basename(log_file).split(".")[0]
            shutil.copy(osp.join(com_dir, f"{mol_id}.com"), dst_dir)


if __name__ == '__main__':
    move_failed_mol_coms("../sol_data/raw/openchem_logP_logs",
                         "../sol_data/raw/openchem_logP_mmff_coms",
                         "../sol_data/raw/openchem_logP_mmff_coms_round1")
