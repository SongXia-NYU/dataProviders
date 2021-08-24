from typing import Union

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
                 prop_dict_raw: dict = None, gauss_version: int = 16):
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
        self.gauss_version = gauss_version
        self._dipole = dipole
        from mendeleev import get_all_elements
        self._element_dict = {e.symbol: e.atomic_number for e in get_all_elements()}
        # self._element_dict = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Se': 34,
        #                       'Br': 35, 'I': 53}
        # for conversion of element_p, which is the column and period of element
        self._element_periodic = {e.symbol: [e.period, e.group.group_id if e.group is not None else None]
                                  for e in get_all_elements()}
        # self._element_periodic = {"H": [1, 1], "B": [2, 3], "C": [2, 4], "N": [2, 5], "O": [2, 6], "F": [2, 7],
        #                           "P": [3, 5], "S": [3, 6], "Cl": [3, 7], "Br": [4, 7], "I": [5, 17], 'Se': [4, 16]}

        self.log_path = log_path
        self.log_lines = open(log_path).readlines() if log_path is not None else None
        self.normal_termination = self._normal_finishes() if qm_sdf is None else True
        if not self.normal_termination:
            print("{} did not terminate normally!!".format(log_path))
            return

        self.base_name = osp.basename(log_path).split(".")[0] \
            if log_path is not None else osp.basename(qm_sdf).split(".")[0]
        self.dir = osp.dirname(log_path) if log_path is not None else osp.dirname(qm_sdf)

        self.mmff_sdf = mmff_sdf
        self.mmff_lines = open(mmff_sdf).readlines() if mmff_sdf is not None else None
        if qm_sdf is None:
            qm_sdf = osp.join(self.dir, self.base_name + ".qm.sdf")
            if (not osp.exists(qm_sdf)) and (log_path is not None):
                os.system("obabel -ig16 {} -osdf -O {}".format(log_path, qm_sdf))
        self.qm_sdf = qm_sdf
        self.qm_lines = open(qm_sdf).readlines()

        self.hartree2ev = Hartree / eV

        self.prop_dict_raw = prop_dict_raw
        if self.prop_dict_raw is None:
            self._reference = np.load(osp.join(osp.dirname(__file__), "atomref.B3LYP_631Gd.10As.npz"))["atom_ref"]
            self.prop_dict_raw = {}
            # read properties from log file
            self._read_prop()
        # get elements from .sdf file
        self._get_elements()
        # get coordinates of the elements from .sdf file
        self._get_coordinates()
        self._charges_mulliken = None
        if log_path is not None:
            # subtract reference energy
            self._prop_ref()

    def _read_prop(self):
        """
        Migrated from Jianing's Frag20_prepare:
        https://github.com/jenniening/Frag20_prepare/blob/master/DataGen/prepare_data.py
        :return:
        """
        self.prop_dict_raw["f_name"] = self.base_name
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
        end_list = ["Normal", "termination", "of"] if self.gauss_version == 16 else ["Job", "finishes", "at:"]

        for line in self.log_lines[-10:]:
            if line.split()[0:3] == end_list:
                return True
        return False

    def _get_elements(self):
        """ Get elements infor for both atomic number, and periodic based """
        def _get_atoms(lines):
            _atoms = []
            natoms = 0
            for line in lines[4:]:
                if len(line.strip().split()) != 16:
                    break
                else:
                    _atoms.append(line)
                    natoms += 1
            return natoms, _atoms

        QMnatoms, atoms = _get_atoms(self.qm_lines)
        if self.mmff_lines is not None:
            MMFFnatoms, _ = _get_atoms(self.mmff_lines)
            assert QMnatoms == MMFFnatoms, "Error: different number of atoms in mmff and qm optimized files"
        self.n_atoms = QMnatoms
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

    @property
    def charges_mulliken(self):
        """ Get Mulliken charges """
        if self._charges_mulliken is None:
            if self.log_lines is None:
                return None
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
            self._charges_mulliken = charges
        return self._charges_mulliken

    @property
    def dipole(self):
        """ Calculate dipole using coordinates and charge for each atom """
        if self._dipole is None:
            if self.charges_mulliken is None:
                return None
            coords = self.qm_coords
            dipole = [[coords[i][0] * self.charges_mulliken[i], coords[i][1] * self.charges_mulliken[i],
                       coords[i][2] * self.charges_mulliken[i]] for i in range(self.n_atoms)]
            dipole = np.sum(dipole, axis=0)
            self._dipole = dipole
        return self._dipole

    def _prop_ref(self):
        """ Get properties for each molecule, and convert properties in Hartree unit into eV unit """
        if "U0" in self.prop_dict_raw.keys():
            reference_total_U0 = np.sum([self._reference[i][1] for i in self.elements])
            self.prop_dict_raw["U0_atom"] = (self.prop_dict_raw["U0"] - reference_total_U0)
        if "U" in self.prop_dict_raw.keys():
            reference_total_U = np.sum([self._reference[i][2] for i in self.elements])
            self.prop_dict_raw["U_atom"] = (self.prop_dict_raw["U"] - reference_total_U)
        if "H" in self.prop_dict_raw.keys():
            reference_total_H = np.sum([self._reference[i][3] for i in self.elements])
            self.prop_dict_raw["H_atom"] = (self.prop_dict_raw["H"] - reference_total_H)
        if "G" in self.prop_dict_raw.keys():
            reference_total_G = np.sum([self._reference[i][4] for i in self.elements])
            self.prop_dict_raw["G_atom"] = (self.prop_dict_raw["G"] - reference_total_G)

    def get_data_frame(self) -> pd.DataFrame:
        """
        :return: a single line dataframe in eV unit
        """
        pd_dict = {key: [self.prop_dict_raw[key]] for key in self.prop_dict_raw}
        return pd.DataFrame(pd_dict)

    def get_error_lines(self) -> pd.DataFrame:
        lines_track = 10
        error_lines = {"f_name": osp.basename(self.log_path)}
        for i in range(1, lines_track+1):
            error_lines[f"error_line_-{i}"] = [self.log_lines[-i]]
        return pd.DataFrame(error_lines)

    def get_torch_data(self) -> torch_geometric.data.Data:
        _tmp_data = {"R": torch.as_tensor(self.qm_coords).view(-1, 3),
                     "Z": torch.as_tensor(self.elements).view(-1),
                     "Q": torch.as_tensor([0.]).view(-1),
                     "F": torch.as_tensor([[0., 0., 0.]]).view(-1, 3),
                     "N": torch.as_tensor(self.n_atoms).view(-1)}
        if self.dipole is not None:
            _tmp_data["D"] = torch.as_tensor(self.dipole).view(-1, 3)
        if self.charges_mulliken is not None:
            _tmp_data["Z_atom"] = torch.as_tensor(self.charges_mulliken).double().view(-1)
        for key in self.prop_dict_raw:
            if key == "U0_atom":
                _tmp_data["E"] = torch.as_tensor(self.prop_dict_raw["U0_atom"]).view(-1)
            elif key == "dd_target":
                _tmp_data.update(self.prop_dict_raw["dd_target"])
            else:
                data = self.prop_dict_raw[key]
                if isinstance(data, str):
                    _tmp_data[key] = data
                elif isinstance(data, int):
                    _tmp_data[key] = torch.as_tensor(data).long()
                elif isinstance(data, float):
                    _tmp_data[key] = torch.as_tensor(data).double()
                elif isinstance(data, torch.Tensor):
                    _tmp_data[key] = data
                else:
                    pass
        return Data(**_tmp_data)


def read_gauss_log(input_file, output_path, indexes=None, gauss_version=16):
    if indexes is not None:
        log_files = [input_file.format(i) for i in indexes]
    else:
        log_files = glob(input_file)
    result_df = pd.DataFrame()
    error_df = pd.DataFrame()
    data_list = []
    for log_file in tqdm(log_files):
        info = Gauss16Info(log_path=log_file, gauss_version=gauss_version)
        if info.normal_termination:
            result_df = result_df.append(info.get_data_frame())
            data_list.append(info.get_torch_data())
        else:
            error_df = error_df.append(info.get_error_lines())

    if osp.isdir(output_path):
        result_df.to_csv(osp.join(output_path, "out.csv"), index=False)
        error_df.to_csv(osp.join(output_path, "error.xlsx"), index=False)
        torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), "out.pt")
    else:
        result_df.to_csv(output_path + ".csv", index=False)
        error_df.to_csv(output_path + "_error.xlsx", index=False)
        torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), output_path + ".pt")


def preprocess_frag20_sol():
    geometry = "mmff_gen"
    dd_csv_folder = "/scratch/projects/yzlab/group/temp_dd/solvation/calculated/"
    train_csv = pd.read_csv(osp.join(dd_csv_folder, "all.csv"))
    valid_csv = pd.read_csv(osp.join(dd_csv_folder, "valid.csv"))
    test_csv = pd.read_csv(osp.join(dd_csv_folder, "test.csv"))
    # concatenate them in this order
    concat_csv = pd.concat([train_csv, valid_csv, test_csv], ignore_index=True)

    jl_root = "/ext3"
    extra_info_heavy = {i: torch.load(osp.join(jl_root, "Frag20_{}_extra_target.pt".format(i))) for i in range(9, 21)}
    tgt_info_heavy = {i: pd.read_csv(osp.join(jl_root, "Frag20_{}_target.csv".format(i)))
                      for i in range(9, 21)}
    # different naming for different geometries
    frag20_ext = ".opt" if geometry == "qm" else ""
    cccd_ext = ".opt" if geometry == "qm" else "_min"

    ccdc_root = "/scratch/sx801/data/CSD20/CSD20/CSD20_data"
    ccdc_extra_target = torch.load("/ext3/CSD20_extra_target.pt")
    ccdc_target = pd.read_csv("/ext3/CSD20_target.csv")

    save_root = "/scratch/sx801/data/Frag20-Sol"
    os.makedirs(save_root, exist_ok=True)

    data_list = []
    success_map = []
    for i in tqdm(range(concat_csv.shape[0])):
        this_id = int(concat_csv["ID"].iloc[i])
        this_source = concat_csv["SourceFile"].iloc[i]
        if geometry in ["qm", "mmff", "mmff_gen"]:
            if this_source == "ccdc":
                mask = (ccdc_target["index"] == this_id).values.reshape(-1)
                tgt_dict = ccdc_target.loc[mask].iloc[0].to_dict()
                sdf_file = osp.join(ccdc_root, "{}{}.sdf".format(this_id, cccd_ext))
                dipole = ccdc_extra_target["dipole"][mask]
            else:
                n_heavy = 9 if this_source == "less10" else int(this_source)
                mask = (tgt_info_heavy[n_heavy]["index"] == this_id).values.reshape(-1)
                tgt_dict = tgt_info_heavy[n_heavy].loc[mask].iloc[0].to_dict()
                if n_heavy > 9:
                    sdf_file = osp.join(jl_root, "Frag20_{}_data".format(n_heavy),
                                        "{}{}.sdf".format(this_id, frag20_ext))
                else:
                    sdf_file = osp.join(jl_root, "Frag20_{}_data".format(n_heavy), "pubchem",
                                        "{}{}.sdf".format(this_id, frag20_ext))
                dipole = extra_info_heavy[n_heavy]["dipole"][mask]
        else:
            raise ValueError("invalid geometry: " + geometry)

        if geometry == "mmff_gen":
            sdf_file = osp.join("/ext3/mmff_sdfs/{}.mmff.sdf".format(i))

        if not osp.exists(sdf_file):
            success_map.append(0)
            continue

        tmp = {}
        for name in ["gasEnergy", "watEnergy", "octEnergy", "CalcSol", "CalcOct", "calcLogP"]:
            tmp[name] = torch.as_tensor(concat_csv[name].iloc[i]).view(-1)
        tgt_dict["dd_target"] = tmp

        this_info = Gauss16Info(qm_sdf=sdf_file, dipole=dipole, prop_dict_raw=tgt_dict)

        data = this_info.get_torch_data()
        data_edge = my_pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                     cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                     bond_atom_sep=False, record_long_range=True)
        data_list.append(data_edge)

        success_map.append(1)

    print("collating and saving...")
    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               osp.join(save_root, "frag20_sol_{}_cutoff-10.pt".format(geometry)))

    success_map = torch.as_tensor(success_map).long().view(-1)
    print("Success: {}/{}".format(success_map.sum(), success_map.shape[0]))
    torch.save(success_map, "success_map.pt")

    train_size = train_csv.shape[0]
    valid_size = valid_csv.shape[0]
    test_size = test_csv.shape[0]
    torch.save({"train_index": torch.arange(train_size),
                "valid_index": torch.arange(train_size, train_size + valid_size),
                "test_index": torch.arange(train_size + valid_size, train_size + valid_size + test_size)},
               osp.join(save_root, "frag20_sol_split_{}_03222021.pt".format(geometry)))


def sdf_to_pt(n_heavy, src_root, dst_root, geometry="qm"):
    """
    Preprocess Frag20 dataset into PyTorch geometric format
    :param n_heavy:
    :param src_root:
    :param dst_root:
    :param geometry:
    :return:
    """
    data_list = []

    target_csv_f = osp.join(src_root, "Frag20_{}_target.csv".format(n_heavy, n_heavy))
    extra_target_f = osp.join(src_root, "Frag20_{}_extra_target.pt".format(n_heavy))
    extra_target = torch.load(extra_target_f)
    target_csv = pd.read_csv(target_csv_f)

    _f_name = ".opt" if geometry == "qm" else ""

    if n_heavy >= 10:
        indexes = target_csv["index"].values.reshape(-1).tolist()
        sdf = [osp.join(src_root, "Frag20_{}_data".format(n_heavy), "{}{}.sdf".format(i, _f_name)) for i in indexes]
    else:
        index_csv = pd.read_csv(osp.join(src_root, "Frag20_{}_index.csv".format(n_heavy, n_heavy)))
        indexes = index_csv["idx"].values.reshape(-1).tolist()
        sources = index_csv["source"].values.reshape(-1).tolist()
        sdf = [osp.join(src_root, "Frag20_{}_data".format(n_heavy), "{}".format(s), "{}{}.sdf".format(i, _f_name))
               for i, s in zip(indexes, sources)]

    for i in tqdm(range(target_csv.shape[0]), "processing heavy: {}".format(n_heavy)):
        this_info = Gauss16Info(qm_sdf=sdf[i], dipole=extra_target["dipole"][i],
                                prop_dict_raw=target_csv.iloc[i].to_dict())
        data = this_info.get_torch_data()
        data_edge = my_pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                     cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                     bond_atom_sep=False, record_long_range=True)
        data_list.append(data_edge)

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               osp.join(dst_root, "frag20_{}_{}_raw.pt".format(n_heavy, geometry)))


def sdf_to_pt_eMol9(src_root, dst_root, geometry="qm"):
    sdf_to_pt_custom(src_root, dst_root, "eMol9", geometry=geometry)


def sdf_to_pt_custom(src_root, dst_root, dataset_name, geometry="qm"):
    target_csv_f = osp.join(src_root, "{}_target.csv".format(dataset_name))
    extra_target_f = osp.join(src_root, "{}_extra_target.pt".format(dataset_name))
    extra_target = torch.load(extra_target_f)
    target_csv = pd.read_csv(target_csv_f)
    indexes = target_csv["f_name"].values.reshape(-1).tolist()
    sdf = [osp.join(src_root, "{}_data".format(dataset_name), "{}.{}.sdf".format(i, geometry)) for i in indexes]

    data_list = []
    for i in tqdm(range(target_csv.shape[0])):
        this_info = Gauss16Info(qm_sdf=sdf[i], dipole=extra_target["dipole"][i],
                                prop_dict_raw=target_csv.iloc[i].to_dict())
        data = this_info.get_torch_data()
        data_edge = my_pre_transform(data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                     cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                     bond_atom_sep=False, record_long_range=True)
        data_list.append(data_edge)
    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               osp.join(dst_root, "{}_raw_{}.pt".format(dataset_name, geometry)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--gauss_version", type=int, default=16)
    args = parser.parse_args()
    # if args.index_file is not None:
    #     _indexes = pd.read_csv(args.index_file)["index"].values.reshape(-1).tolist()
    # else:
    #     _indexes = None
    # read_gauss_log(args.input_file, args.output_file, _indexes, args.gauss_version)
    preprocess_frag20_sol()
