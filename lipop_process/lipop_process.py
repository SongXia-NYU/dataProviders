import os
import shutil

from GaussUtils.GaussInfo import read_gauss_log
from GaussUtils.combine_3phases_data import infuse_energy, combine_csv

if __name__ == '__main__':
    # read_gauss_log("../sol_data/raw/lipop_logs/*.log", "lipop")
<<<<<<< HEAD
    # read_gauss_log("../sol_data/raw/lipop_water_coms/*.log", "lipop_water")
    read_gauss_log("../sol_data/raw/lipop_oct_coms/*.log", "lipop_oct")
=======
    # read_gauss_log("../sol_data/raw/lipop_water_coms/*.log", "lipop_water", test_run=False)
    # read_gauss_log("../sol_data/raw/lipop_oct_coms/*.log", "lipop_oct")
>>>>>>> 7c954564772de1823fa5efc0a919157a7bcc3acb
    # combine_csv("lipop.csv", "lipop_water.csv", "lipop_oct.csv", "lipop_paper.csv", "lipop_sol.csv")
    # os.makedirs("./processed", exist_ok=True)
    # shutil.move("lipop.pt", "processed/lipop.pt")
    infuse_energy("lipop_sol.csv", "lipop.pt", ".")
