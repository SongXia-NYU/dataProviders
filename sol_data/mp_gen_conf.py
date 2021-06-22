from DataGen.genconfs import runGenerator
from multiprocessing import Pool
import pandas as pd


concat_csv = pd.read_csv("lipop.csv")


def _run_generator(i):
    runGenerator([concat_csv.index.tolist()[i]], [concat_csv["cano_smiles"].tolist()[i]], "lipop", "raw/lipop_confs")


if __name__ == '__main__':
    with Pool(16) as p:
        p.map(_run_generator, range(concat_csv.shape[0]))
