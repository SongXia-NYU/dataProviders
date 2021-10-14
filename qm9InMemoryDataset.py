import os

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset

from DataPrepareUtils import my_pre_transform, name_extender, physnet_to_datalist


class Qm9InMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, boundary_factor=100., extended_bond=False,
                 edge_version='cutoff', cutoff=10., cal_3body_term=True, sort_edge=True, use_center=True,
                 bond_atom_sep=True, cal_efg=False, record_long_range=False, type_3_body='B', geometry='QM',
                 split_file="split_qm9.npz"):
        """

        :param root:
        :param transform:
        :param pre_transform:
        :param boundary_factor: Manually setup a boundary for Voronoi Diagram
        :param edge_version: when 'voronoi' use Voronoi Diagram to calculate edge; when 'cutoff', use cutoff as edge
        :param cutoff: only required when edge_version=='cutoff'
        :param cal_3body_term: calculate edge-edge interaction
        :param sort_edge: sort edge to speed up scatter_ when training
        :param use_center: center molecule when doing voronoi Diagram
        :param bond_atom_sep: If true, use defined bond when calculating bond-bond interaction, atom-atom interaction
                              remains the same.
        :param cal_efg: Calculate efg related data, deprecated_code since it gives no improvement
        """
        self.geometry = geometry
        self.extended_bond = extended_bond
        self.type_3_body = type_3_body
        self.record_long_range = record_long_range
        self.cal_efg = cal_efg
        self.bond_atom_sep = bond_atom_sep
        self.debug_mode = True
        self.boundary_factor = boundary_factor
        self.edge_version = edge_version
        self.cutoff = cutoff
        self.cal_3body_term = cal_3body_term
        self.sort_edge = sort_edge
        self.use_center = use_center
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_index, self.val_index, self.test_index = self.get_split_indexes(split_file)

    def get_split_indexes(self, split_file):
        shuffle_index_file = os.path.join(self.processed_dir, split_file)
        if os.path.exists(shuffle_index_file):
            shuffle_index = np.load(shuffle_index_file)
            train_index, val_index, test_index = shuffle_index['train'], shuffle_index['validation'], shuffle_index[
                'test']
            return train_index, val_index, test_index
        else:
            raise ValueError('Could not find QM9 split file! Exiting...')

    @property
    def raw_file_names(self):
        """
        record id to better idx_name paths
        :return:
        """
        _id = 0
        raw_files = ['qm9_{}_removeproblem.npz'.format(self.geometry.lower())]
        self.main_id = _id
        if self.cal_efg:
            raw_files.append('EFGs_QM9.pt')
            _id += 1
            self.EFG_data_id = _id
        if self.bond_atom_sep:
            raw_files.append('QM9SDFs_removeproblem.pt')
            _id += 1
            self.SDF_data_id = _id
        return raw_files

    @property
    def processed_file_names(self):
        name = 'qm9'

        name = name_extender(name, cal_3body_term=self.cal_3body_term, edge_version=self.edge_version,
                             cutoff=self.cutoff, boundary_factor=self.boundary_factor, extended_bond=self.extended_bond,
                             use_center=self.use_center, bond_atom_sep=self.bond_atom_sep, geometry=self.geometry,
                             record_long_range=self.record_long_range, type_3_body=self.type_3_body)

        return name

    def download(self):
        raise FileNotFoundError('Can not find raw file(s)! exiting...')

    def process(self):
        qm9_raw_data = np.load(self.raw_paths[self.main_id])
        N = torch.LongTensor(qm9_raw_data['N'])
        R = torch.DoubleTensor(qm9_raw_data['R'])
        E = torch.DoubleTensor(qm9_raw_data['E'])
        Q = torch.DoubleTensor(qm9_raw_data['Q'])
        D = torch.DoubleTensor(qm9_raw_data['D'])
        Z = torch.LongTensor(qm9_raw_data['Z'])
        num_mol = qm9_raw_data['N'].shape[0]

        if self.cal_efg:
            '''
            Load EFGs related data
            '''
            efg_data = torch.load(self.raw_paths[self.EFG_data_id])
            EFG_R = torch.DoubleTensor(efg_data['efgs_mass_center'])
            EFG_Z = torch.LongTensor(efg_data['efg_types'])
            num_efg = torch.LongTensor(efg_data['num_efgs'])
            efgs_batch = torch.LongTensor(efg_data['efgs_batch'])
        else:
            EFG_R, EFG_Z, num_efg, efgs_batch = None, None, None, None

        if self.bond_atom_sep:
            '''
            Load mol data to calculate bonding edge
            '''
            mols = torch.load(self.raw_paths[self.SDF_data_id])
        else:
            mols = None

        data_list = physnet_to_datalist(self, N, R, E, D, Q, Z, num_mol, mols, efgs_batch, EFG_R, EFG_Z, num_efg)
        print('collating...')
        data, slices = self.collate(data_list)
        print('saving...')
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    qm9Data = Qm9InMemoryDataset(root='data', pre_transform=my_pre_transform, record_long_range=True,
                                 extended_bond=True)
    # pair_dist = torch.nn.PairwiseDistance()

    # def cal_edge_length(R, edge_index, dist_calculator):
    #     index_1 = edge_index[0, :]
    #     r_1 = R[index_1]
    #     index_2 = edge_index[1, :]
    #     r_2 = R[index_2]
    #     return dist_calculator(r_1, r_2).view(-1)
    #
    #
    # import matplotlib.pyplot as plt
    #
    # non_edge_length = [cal_edge_length(data.R, data.atom_edge_index, pair_dist) for data in qm9Data]
    # bond_edge_length = [cal_edge_length(data.R, data.bonding_edge_index, pair_dist) for data in qm9Data]
    # non_edge_length = torch.cat(non_edge_length)
    # bond_edge_length = torch.cat(bond_edge_length)
    #
    # plt.hist(non_edge_length, 200, label='non-bonding edge')
    # plt.hist(bond_edge_length, 25, label='bonding-edge')
    # plt.legend()
    # plt.xlabel('Distance/A')
    # plt.ylabel('Count')
    # plt.title('Edge distance distribution')
    # current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # plt.savefig('../figures/fig_' + current_time)
    # plt.show()

    print('Finished')
