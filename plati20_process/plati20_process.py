import os
import shutil

from GaussUtils.GaussInfo import read_gauss_log
from GaussUtils.combine_3phases_data import infuse_energy, combine_csv

if __name__ == '__main__':
    read_gauss_log("../sol_data/raw/plati20_gas_coms/*.log", "plati20_gas")
    read_gauss_log("../sol_data/raw/plati20_water_coms/*.log", "plati20_water")
    read_gauss_log("../sol_data/raw/plati20_oct_coms/*.log", "plati20_oct")
    # combine_csv("plati20.csv", "plati20_water.csv", "plati20_oct.csv", "plati20_paper.csv", "plati20_sol.csv")
    # os.makedirs("./processed", exist_ok=True)
    # shutil.move("plati20.pt", "processed/plati20.pt")
    # infuse_energy("plati20_sol.csv", "plati20.pt", ".")
