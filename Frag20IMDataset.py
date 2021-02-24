import os
import time

import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data

from DataPrepareUtils import my_pre_transform, name_extender, physnet_to_datalist


class Frag20IMDataset(InMemoryDataset):
    def __init__(self, root, n_heavy_atom, transform=None, pre_transform=None, boundary_factor=100.,
                 extended_bond=False,
                 edge_version='cutoff', cutoff=10., cal_3body_term=True, sort_edge=True, use_center=True,
                 bond_atom_sep=True, cal_efg=False, record_long_range=False, type_3_body='B', geometry='QM',
                 add_sol=False):
        """
        This dataset contains one single number of heavy atoms
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
        :param cal_efg: Calculate efg related data, deprecated_code because of no improvement to training
        """
        self.add_sol = add_sol
        self.geometry = geometry
        self.extended_bond = extended_bond
        self.n_heavy_atom = n_heavy_atom
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
        self.train_index, self.val_index, self.test_index = self.get_split_indexes()

    def get_split_indexes(self):
        shuffle_index_file = os.path.join(self.processed_dir, 'split_frag9.pt')
        if os.path.exists(shuffle_index_file):
            print('Loading existing split file...')
            shuffle_index = torch.load(shuffle_index_file)
            train_index, val_index, test_index = shuffle_index['train'], shuffle_index['validation'], shuffle_index[
                'test']
            print('loading split file done')
        else:
            print('No exist split file, will random generate it.')
            data_size = len(self)
            _shuffled = torch.randperm(data_size)
            train_index, val_index, test_index = _shuffled[:99000], _shuffled[99000:100000], _shuffled[100000:101000]
            torch.save({'train': train_index, 'validation': val_index, 'test': test_index}, shuffle_index_file)
            print('successfully generated and saved to dir.')

        return train_index, val_index, test_index

    @property
    def raw_file_names(self):
        if self.n_heavy_atom == 9:
            names = ['Frag9_loose.npz', 'Index_strict.csv', 'Frag9_strict_mols.pt']
            names = [osp.join('frag20', name) for name in names]
        else:
            names = ['Frag20_{}_PhysNet_{}.npz'.format(self.n_heavy_atom, self.geometry),
                     'Frag20_{}_sdf_{}.pt'.format(self.n_heavy_atom, self.geometry)]
            names = [osp.join('frag20', name) for name in names]

        names.append(osp.join("frag20", "frag20_solvation.csv"))
        return names

    @property
    def processed_file_names(self):
        name = 'frag20_{}'.format(self.n_heavy_atom)

        if self.add_sol:
            name += "Sol"

        name = name_extender(name, cal_3body_term=self.cal_3body_term, edge_version=self.edge_version,
                             cutoff=self.cutoff, boundary_factor=self.boundary_factor, extended_bond=self.extended_bond,
                             record_long_range=self.record_long_range, type_3_body=self.type_3_body,
                             use_center=self.use_center, bond_atom_sep=self.bond_atom_sep, geometry=self.geometry)
        return name

    def download(self):
        print('Download method not implemented')
        print(self.raw_file_names)
        raise ValueError('Please make sure you have raw data files in data/raw/ folder')

    def process(self):
        frag20_raw_data = np.load(self.raw_paths[0])
        if self.n_heavy_atom == 9:
            index_data = pd.read_csv(self.raw_paths[1])
            strict_index = torch.as_tensor(index_data['index_in_indexbaseline']).long()
            N = torch.as_tensor(frag20_raw_data['N'])[strict_index].long()
            R = torch.as_tensor(frag20_raw_data['R_{}'.format(self.geometry.lower())])[strict_index, :, :].double()
            E = torch.as_tensor(frag20_raw_data['E'])[strict_index].double()
            Q = torch.as_tensor(frag20_raw_data['Q'])[strict_index].double()
            D = torch.as_tensor(frag20_raw_data['D_qm'])[strict_index].double()
            Z = torch.as_tensor(frag20_raw_data['Z'])[strict_index, :].long()
        else:
            N = torch.as_tensor(frag20_raw_data['N'])[:].long()
            R = torch.as_tensor(frag20_raw_data['R'])[:, :, :].double()
            E = torch.as_tensor(frag20_raw_data['E'])[:].double()
            Q = torch.as_tensor(frag20_raw_data['Q'])[:].double()
            D = torch.as_tensor(frag20_raw_data['D'])[:].double()
            Z = torch.as_tensor(frag20_raw_data['Z'])[:, :].long()
        num_mol = N.shape[0]

        if self.cal_efg:
            '''
            Load EFGs related data
            '''
            if len(self.raw_paths) < 4:
                print('You dont have EFGs data generated! Exiting...')
                exit()
            efg_data = torch.load(self.raw_paths[3])
            EFG_R = torch.as_tensor(efg_data['efgs_mass_center']).double()
            EFG_Z = torch.as_tensor(efg_data['efg_types']).long()
            num_efg = torch.as_tensor(efg_data['num_efgs']).long()
            efgs_batch = torch.as_tensor(efg_data['efgs_batch']).long()
        else:
            efgs_batch, EFG_R, EFG_Z, num_efg = None, None, None, None

        if self.bond_atom_sep:
            '''
            Load mol data to calculate bonding edge
            '''
            # print('MOLs currently unavailable!!')
            # mols = None
            mols = torch.load(self.raw_paths[-2])
        else:
            mols = None

        sol_data = pd.read_csv(self.raw_paths[-1])

        data_list = physnet_to_datalist(self, N, R, E, D, Q, Z, num_mol, mols, efgs_batch, EFG_R, EFG_Z, num_efg,
                                        sol_data)
        print('collating...')
        data, slices = self.collate(data_list)
        print('saving...')
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # _data = Frag20IMDataset(root='data', n_heavy_atom=20, pre_transform=my_pre_transform, record_long_range=True,
    #                         bond_atom_sep=True, cal_3body_term=True, geometry='QM', add_sol=True)
    for i in range(9, 21):
        _data = Frag20IMDataset(root='data', n_heavy_atom=i, pre_transform=my_pre_transform, record_long_range=True,
                                bond_atom_sep=True, cal_3body_term=True, geometry='QM', add_sol=True)
    print("hello")
