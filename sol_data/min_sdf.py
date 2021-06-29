import rdkit
from rdkit.Chem import SDMolSupplier, SDWriter
from glob import glob
import numpy as np
import os
import os.path as osp

from tqdm import tqdm


def min_sdf():
    files = glob("raw/openchem_logP_confs/*.sdf")
    for f in tqdm(files):
        try:
            suppl = SDMolSupplier(f)
            lowest_e = np.inf
            selected_mol = None
            for mol in suppl:
                energy = float(mol.GetProp("energy_abs"))
                if energy < lowest_e:
                    lowest_e = energy
                    selected_mol = mol
            if selected_mol is not None:
                writer = SDWriter(f"raw/openchem_logP_mmff_sdfs/{osp.basename(f).split('.')[0].split('_')[0]}.mmff.sdf")
                writer.write(selected_mol)
        except OSError as e:
            print(e)
        except KeyError as e:
            print(e)


if __name__ == "__main__":
    min_sdf()

