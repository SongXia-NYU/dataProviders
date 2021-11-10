from DataGen.genconfs import runGenerator
from multiprocessing import Pool
import pandas as pd


concat_csv = pd.read_csv("BBBP.csv")


def _run_generator(i):
    runGenerator([concat_csv.idx_name.tolist()[i]], [concat_csv["SMILES"].tolist()[i]], "BBBP",
                 "raw/BBBP_confs")


if __name__ == '__main__':
    with Pool(16) as p:
        p.map(_run_generator, range(0, concat_csv.shape[0]))
