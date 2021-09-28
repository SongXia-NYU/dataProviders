import os
import os.path as osp
import shutil

from GaussUtils.GaussInfo import read_gauss_log
from GaussUtils.combine_3phases_data import infuse_energy, combine_csv

if __name__ == '__main__':
    # read_gauss_log("../sol_data/raw/csd20_gas_coms_small/*.log", "csd20_gas_small", cpus=1)
    read_gauss_log("../sol_data/raw/csd20_gas_coms/*.log", "csd20_gas", cpus=32)
    read_gauss_log("../sol_data/raw/csd20_water_coms/*.log", "csd20_water", cpus=32)
    read_gauss_log("../sol_data/raw/csd20_oct_coms/*.log", "csd20_oct", cpus=32)
    # combine_csv("csd20_gas.csv", "csd20_water.csv", "csd20_oct.csv", None, "csd20_sol.csv")
    # os.makedirs("./processed", exist_ok=True)
    # if not osp.exists("processed/csd20_gas.pt"):
    #     shutil.move("csd20_gas.pt", "processed/csd20_gas.pt")
    # infuse_energy("csd20_sol.csv", "csd20_gas.pt", ".")
