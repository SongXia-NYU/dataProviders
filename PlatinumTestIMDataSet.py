import torch
import numpy as np
import pandas as pd

from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from DataPrepareUtils import my_pre_transform, name_extender
from rdkit.Chem import MolFromSmiles, AddHs


def _get_ith_data(data_index, E, N, R, D, Q, Z):
    num_atoms = N[data_index].item()
    _tmp_data = Data()
    _tmp_data.E = E[data_index].view(-1)
    _tmp_data.N = N[data_index].view(-1)
    _tmp_data.R = R[data_index, :num_atoms, :].view(-1, 3)
    _tmp_data.D = D[data_index, :].view(-1, 3)
    _tmp_data.Q = Q[data_index].view(-1)
    _tmp_data.Z = Z[data_index, :num_atoms].view(-1)
    return _tmp_data


class PlatinumTestIMDataSet(InMemoryDataset):
    @property
    def raw_file_names(self):
        if not self.sep_heavy_atom:
            return ['platinum_10_13.npz', 'platinum_14_20.npz']
        else:
            return ['platinum_10_13.npz', 'platinum_14_20.npz', 'RMSD_addNumberHA_10_13.csv',
                    'RMSD_addNumberHA_14_20.csv']

    @property
    def processed_file_names(self):
        name = 'platinum_10_20_qm' if self.qm else 'platinum_10_20_mmff'
        name = name_extender(name, self.cal_3body_term, self.edge_version, self.cutoff, self.boundary_factor,
                             self.use_center, self.bond_atom_sep, self.record_long_range)
        name = name[:-3]
        if self.sep_heavy_atom:
            name = name + '-heavy-{}'.format(self.num_heavy_atom)
        name = name + '.pt'
        return name

    def download(self):
        print('ERROR: You should NOT see this message. Check if you have all of your raw files.')

    def process(self):
        data1 = np.load(self.raw_paths[0])
        data2 = np.load(self.raw_paths[1])
        data1_feed_dict = {
            'E': torch.as_tensor(data1['E']),
            'N': torch.as_tensor(data1['N']),
            'R': torch.as_tensor(data1['R_qm'] if self.qm else data1['R_mmff']),
            'D': torch.as_tensor(data1['D_qm'] if self.qm else data1['D_mmff']),
            'Q': torch.as_tensor(data1['Q']),
            'Z': torch.as_tensor(data1['Z'])
        }
        data2_feed_dict = {
            'E': torch.as_tensor(data2['E']),
            'N': torch.as_tensor(data2['N']),
            'R': torch.as_tensor(data2['R_qm'] if self.qm else data2['R_mmff']),
            'D': torch.as_tensor(data2['D_qm'] if self.qm else data2['D_mmff']),
            'Q': torch.as_tensor(data2['Q']),
            'Z': torch.as_tensor(data2['Z'])
        }

        data1_size = data1['E'].shape[0]
        data2_size = data2['E'].shape[0]

        if not self.sep_heavy_atom:
            data_size = data1_size + data2_size
        else:
            in_part1 = (self.num_heavy_atom < 14)
            heavy_atom_data = pd.read_csv(self.raw_paths[2] if in_part1 else self.raw_paths[3])
            num_heavy_atom = torch.as_tensor(heavy_atom_data['numberHA']).long()
            atom_mask = (num_heavy_atom == self.num_heavy_atom)
            atom_mask = atom_mask.view(-1)
            data_dict_used = data1_feed_dict if in_part1 else data2_feed_dict
            for key in data_dict_used.keys():
                data_dict_used[key] = data_dict_used[key][atom_mask]
            '''
            Here is a trick to make sure later part only calculate data_dict_used
            '''
            data_size = data_dict_used['E'].shape[0]
            data1_feed_dict = data_dict_used

        data_array = np.empty(data_size, dtype=Data)

        for i in tqdm(range(data_size)):

            data_index = i if i < data1_size else i - data1_size

            if i < data1_size:
                tmp_data = _get_ith_data(data_index, **data1_feed_dict)
            else:
                tmp_data = _get_ith_data(data_index, **data2_feed_dict)
            tmp_data = self.pre_transform(tmp_data, edge_version='cutoff', do_sort_edge=True, cal_efg=False,
                                          cutoff=self.cutoff, boundary_factor=None, use_center=None,
                                          mol=AddHs(MolFromSmiles('C')),
                                          cal_3body_term=self.cal_3body_term, bond_atom_sep=self.bond_atom_sep,
                                          record_long_range=self.record_long_range)
            data_array[i] = tmp_data

        data_list = [data_array[i] for i in range(data_size)]
        print('collating...')
        data1, slices = self.collate(data_list)
        print('saving...')
        torch.save((data1, slices), self.processed_paths[0])

    def __init__(self, root, pre_transform, qm=True, cutoff=10.00, boundary_factor=100., sep_heavy_atom=False,
                 num_heavy_atom=None, use_center=True,
                 record_long_range=False, cal_3body_term=True, bond_atom_sep=True, edge_version='cutoff'):
        """

        :param root:
        :param pre_transform:
        :param qm:
        :param cutoff:
        :param sep_heavy_atom:  if true, you must set num_heavy_atom between 10 to 20
        :param num_heavy_atom:
        """
        self.use_center = use_center
        self.boundary_factor = boundary_factor
        self.edge_version = edge_version
        self.bond_atom_sep = bond_atom_sep
        self.cal_3body_term = cal_3body_term
        self.record_long_range = record_long_range
        self.num_heavy_atom = num_heavy_atom
        self.sep_heavy_atom = sep_heavy_atom
        self.qm = qm
        self.cutoff = cutoff
        print('SUPER IMPORTANT: mol files not available, use CH4 for all of the molecules -> CRAZY!')
        super().__init__(root, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.test_index = torch.arange(len(self))


if __name__ == '__main__':
    test_data = PlatinumTestIMDataSet('data', pre_transform=my_pre_transform, sep_heavy_atom=False, num_heavy_atom=None,
                                      cal_3body_term=False, bond_atom_sep=False, record_long_range=True, qm=False)
    print('Finished!')
