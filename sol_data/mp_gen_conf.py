from DataGen.genconfs import runGenerator
from multiprocess.pool import Pool
import pandas as pd


def _run_generator(i):
    runGenerator([concat_csv.index.tolist()[i]], [concat_csv["cano_smiles"].tolist()[i]], "lipop", "raw/lipop_confs")


if __name__ == '__main__':
    concat_csv = pd.read_csv("lipop.csv")
    with Pool(20) as p:
        p.map(_run_generator, range(concat_csv.shape[0]))
