import pandas as pd
from ase.units import Hartree, eV, Bohr, Ang
import os
import torch
import torch_geometric
from torch_geometric.data import Data
import os.path as osp
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np

from DataPrepareUtils import my_pre_transform


class Gauss16Info:
    def __init__(self, log_path: str = None, qm_sdf: str = None, mmff_sdf: str = None, dipole: torch.Tensor = None,
                 prop_dict_raw: dict = None):
        """
        Extract information from Gaussian 16 log files OR from SDF files. In the later case, qm_sdf, dipole and
        prop_dict_raw must not be None
        :param log_path:
        :param qm_sdf:
        :param mmff_sdf:
        :param dipole:
        :param prop_dict_raw:
        """
        # for conversion of element, which is the atomic number of element
        self.dipole = dipole
        self._element_dict = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35}
        # for conversion of element_p, which is the column and period of element
        self._element_periodic = {"H": [1, 1], "B": [2, 3], "C": [2, 4], "N": [2, 5], "O": [2, 6], "F": [2, 7],
                                  "P": [3, 5], "S": [3, 6], "Cl": [3, 7], "Br": [4, 7]}

        self.log_path = log_path
        self.log_lines = open(log_path).readlines() if log_path is not None else None
        self.normal_termination = self._normal_finishes() if qm_sdf is None else True
        if not self.normal_termination:
            return

        self.base_name = osp.basename(log_path).split(".")[0]\
            if log_path is not None else osp.basename(qm_sdf).split(".")[0]
        self.dir = osp.dirname(log_path) if log_path is not None else osp.dirname(qm_sdf)

        self.mmff_sdf = mmff_sdf
        self.mmff_lines = open(mmff_sdf).readlines() if mmff_sdf is not None else None
        if qm_sdf is None:
            qm_sdf = osp.join(self.dir, self.base_name + ".qm.sdf")
            os.system("obabel -ig16 {} -osdf -O {}".format(log_path, qm_sdf))
        self.qm_sdf = qm_sdf
        self.qm_lines = open(qm_sdf).readlines()

        self.hartree2ev = Hartree / eV

        self.prop_dict_raw = prop_dict_raw
        if self.prop_dict_raw is None:
            self._reference = np.load("atomref.B3LYP_631Gd.10As.npz")["atom_ref"]
            self.prop_dict_raw = {}
            # read properties from log file
            self._read_prop()
        # get elements from .sdf file
        self._get_elements()
        # get coordinates of the elements from .sdf file
        self._get_coordinates()
        if self.dipole is None:
            # get mulliken charge from log file
            self._get_mulliken_charges()
            # calculate dipole
            self._get_dipole()
        if log_path is None:
            # subtract reference energy
            self._prop_ref()

    def _read_prop(self):
        """
        Migrated from Jianing's Frag20_prepare:
        https://github.com/jenniening/Frag20_prepare/blob/master/DataGen/prepare_data.py
        :return:
        """
        for idx, line in enumerate(self.log_lines):
            if line.startswith(' Rotational constants'):
                vals = line.split()
                self.prop_dict_raw['A'] = float(vals[-3])
                self.prop_dict_raw['B'] = float(vals[-2])
                self.prop_dict_raw['C'] = float(vals[-1])
            elif line.startswith(' Dipole moment'):
                self.prop_dict_raw['mu'] = float(self.log_lines[idx + 1].split()[-1])
            elif line.startswith(' Isotropic polarizability'):
                self.prop_dict_raw['alpha'] = float(line.split()[-2])
            elif (line.startswith(' Alpha  occ. eigenvalues') and
                  self.log_lines[idx + 1].startswith(' Alpha virt. eigenvalues')):
                self.prop_dict_raw['ehomo'] = float(self.log_lines[idx + 1].split()[4]) * self.hartree2ev
                self.prop_dict_raw['elumo'] = float(line.split()[-1]) * self.hartree2ev
                self.prop_dict_raw['egap'] = (float(self.prop_dict_raw['ehomo']) - float(
                    self.prop_dict_raw['elumo'])) * self.hartree2ev
            elif line.startswith(' Electronic spatial extent'):
                self.prop_dict_raw['R2'] = float(line.split()[-1])
            elif line.startswith(' Zero-point correction'):
                self.prop_dict_raw['zpve'] = float(line.split()[-2]) * self.hartree2ev
            elif line.startswith(' Sum of electronic and zero-point Energies'):
                self.prop_dict_raw['U0'] = float(line.split()[-1]) * self.hartree2ev
            elif line.startswith(' Sum of electronic and thermal Energies'):
                self.prop_dict_raw['U'] = float(line.split()[-1]) * self.hartree2ev
            elif line.startswith(' Sum of electronic and thermal Enthalpies'):
                self.prop_dict_raw['H'] = float(line.split()[-1]) * self.hartree2ev
            elif line.startswith(' Sum of electronic and thermal Free Energies'):
                self.prop_dict_raw['G'] = float(line.split()[-1]) * self.hartree2ev
            elif line.startswith(' Total       '):
                self.prop_dict_raw['Cv'] = float(line.split()[-2])
            elif line.startswith(' SCF Done'):
                self.prop_dict_raw['E'] = float(line.split()[4]) * self.hartree2ev

    def _normal_finishes(self):
        line = self.log_lines[-1]
        if line.split()[0:3] == ["Normal", "termination", "of"]:
            return True
        else:
            return False

    def _get_elements(self):
        """ Get elements infor for both atomic number, and periodic based """
        QMnatoms = int(self.qm_lines[3].split()[0])
        if self.mmff_lines is not None:
            MMFFnatoms = int(self.mmff_lines[3].split()[0])
            assert QMnatoms == MMFFnatoms, "Error: different number of atoms in mmff and qm optimized files"
        self.n_atoms = QMnatoms
        atoms = self.qm_lines[4:self.n_atoms + 4]
        elements = []
        elements_p = []
        for atom in atoms:
            atom = atom.split()
            elements.append(self._element_dict[atom[3]])
            elements_p.append(self._element_periodic[atom[3]])
        self.elements = elements
        self.elements_p = elements_p

    def _get_coordinates(self):
        """ Get atom coordinates for both MMFF and QM """
        atoms_MMFF = self.mmff_lines[4:self.n_atoms + 4] if self.mmff_lines is not None else None
        atoms_QM = self.qm_lines[4:self.n_atoms + 4]
        positions_QM = []
        for atom in atoms_QM:
            atom = atom.split()
            positions_QM.append([float(pos) for pos in atom[:3]])

        if atoms_MMFF is not None:
            positions_MMFF = []
            for atom in atoms_MMFF:
                atom = atom.split()
                positions_MMFF.append([float(pos) for pos in atom[:3]])
        else:
            positions_MMFF = None

        self.mmff_coords = positions_MMFF
        self.qm_coords = positions_QM

    def _get_mulliken_charges(self):
        """ Get Mulliken charges """
        index = [idx for idx, line in enumerate(self.log_lines) if line.startswith(" Mulliken charges:")][0] + 2
        natoms_old = self.n_atoms
        natoms = self.n_atoms
        try:
            charges = [float(line.split()[-1]) for line in self.log_lines[index: index + natoms]]
        except:
            charges = []
            for idx, line in enumerate(self.log_lines[index:]):
                if idx < natoms:
                    # remove calculation comments in Mulliken charges part ###
                    try:
                        charge = float(line.split()[-1])
                        charges.append(charge)
                    except:
                        print(line)
                        natoms += 1
                        continue
        assert len(charges) == natoms_old, "Error: charges are wrong"
        self.charges_mulliken = charges

    def _get_dipole(self):
        """ Calculate dipole using coordinates and charge for each atom """
        coords = self.qm_coords
        dipole = [[coords[i][0] * self.charges_mulliken[i], coords[i][1] * self.charges_mulliken[i],
                   coords[i][2] * self.charges_mulliken[i]] for i in range(self.n_atoms)]
        dipole = np.sum(dipole, axis=0)
        self.dipole = dipole

    def _prop_ref(self):
        """ Get properties for each molecule, and convert properties in Hartree unit into eV unit """
        reference_total_U0 = np.sum([self._reference[i][1] for i in self.elements])
        reference_total_U = np.sum([self._reference[i][2] for i in self.elements])
        reference_total_H = np.sum([self._reference[i][3] for i in self.elements])
        reference_total_G = np.sum([self._reference[i][4] for i in self.elements])
        self.prop_dict_raw["U0_atom"] = (self.prop_dict_raw["U0"] - reference_total_U0)
        self.prop_dict_raw["U_atom"] = (self.prop_dict_raw["U"] - reference_total_U)
        self.prop_dict_raw["H_atom"] = (self.prop_dict_raw["H"] - reference_total_H)
        self.prop_dict_raw["G_atom"] = (self.prop_dict_raw["G"] - reference_total_G)

    def get_data_frame(self) -> pd.DataFrame:
        """
        :return: a single line dataframe in eV unit
        """
        pd_dict = {key: [self.prop_dict_raw[key]] for key in self.prop_dict_raw}
        pd_dict["f_name"] = [self.base_name]
        return pd.DataFrame(pd_dict)

    def get_torch_data(self) -> torch_geometric.data.Data:
        _tmp_data = {"R": torch.as_tensor(self.qm_coords).view(-1, 3),
                     "Z": torch.as_tensor(self.elements).view(-1),
                     "Q": torch.as_tensor([0]).view(-1),
                     "D": torch.as_tensor(self.dipole).view(-1, 3),
                     "F": torch.as_tensor([[0., 0., 0.]]).view(-1, 3),
                     "E": torch.as_tensor(self.prop_dict_raw["U0_atom"]).view(-1),
                     "N": torch.as_tensor(self.n_atoms).view(-1)}
        return Data(**_tmp_data)


def read_gauss_log(input_file, output_path):
    log_files = glob(input_file)
    result_csv = pd.DataFrame()
    data_list = []
    for log_file in tqdm(log_files):
        info = Gauss16Info(log_file)
        result_csv = result_csv.append(info.get_data_frame())
        data_list.append(info.get_torch_data())

    if osp.isdir(output_path):
        result_csv.to_csv(osp.join(output_path, "out.csv"), index=False)
        torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), "out.pt")
    else:
        result_csv.to_csv(output_path+".csv", index=False)
        torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), output_path+".pt")


def sdf_to_pt(n_heavy, src_root, dst_root):
    # TODO
    data_list = []

    target_csv_f = osp.join(src_root, "Frag20_{}_target.csv".format(n_heavy, n_heavy))
    extra_target_f = osp.join(src_root, "Frag20_{}_extra_target.pt".format(n_heavy))
    extra_target = torch.load(extra_target_f)
    target_csv = pd.read_csv(target_csv_f)
    indexes = target_csv["index"].values.reshape(-1).tolist()
    opt_sdf = [osp.join(src_root, "Frag20_{}_data".format(n_heavy), "{}.opt.sdf".format(i)) for i in indexes]

    for i in tqdm(range(target_csv.shape[0]), "processing heavy: {}".format(n_heavy)):
        this_info = Gauss16Info(qm_sdf=opt_sdf[i], dipole=extra_target["dipole"][i],
                                prop_dict_raw=target_csv.iloc[i].to_dict())
        data = this_info.get_torch_data()
        data_edge = my_pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                     cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                     bond_atom_sep=False, record_long_range=True)
        data_list.append(data_edge)

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               osp.join(dst_root, "frag20_{}_raw.pt".format(n_heavy)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    read_gauss_log(args.input_file, args.output_file)
