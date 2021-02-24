import os.path as osp

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset

from DataPrepareUtils import name_extender, physnet_to_datalist, my_pre_transform


class PhysDimeIMDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, boundary_factor=100., extended_bond=False,
                 edge_version='cutoff', cutoff=10., cal_3body_term=True, sort_edge=True, use_center=True,
                 bond_atom_sep=True, cal_efg=False, record_long_range=False, type_3_body='B',
                 infile_dic=None, processed_prefix='data'):
        """

        :param root:
        :param transform:
        :param pre_transform:
        :param boundary_factor:
        :param extended_bond:
        :param edge_version:
        :param cutoff:
        :param cal_3body_term:
        :param sort_edge:
        :param use_center:
        :param bond_atom_sep:
        :param cal_efg:
        :param record_long_range:
        :param type_3_body:
        :param infile_dic: a dictionary indicates input files, possible keys: 'PhsNet', 'SDF', 'split'. Values:
        str or list of strings containing file names
        :param processed_prefix:
        """
        self.processed_prefix = processed_prefix
        self.infile_dic = infile_dic
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
        self.data, self.slices, split = torch.load(self.processed_paths[0])
        self.train_index, self.val_index, self.test_index = None, None, torch.arange(len(self))

    @property
    def raw_file_names(self):
        if self.infile_dic is not None:
            result = []
            for key in self.infile_dic:
                val = self.infile_dic[key]
                if not isinstance(val, list):
                    val = [val]
                result.extend(val)
            return result
        else:
            return [""]

    @property
    def processed_file_names(self):
        name = name_extender(self.processed_prefix, cal_3body_term=self.cal_3body_term, edge_version=self.edge_version,
                             cutoff=self.cutoff, boundary_factor=self.boundary_factor, extended_bond=self.extended_bond,
                             record_long_range=self.record_long_range, type_3_body=self.type_3_body,
                             use_center=self.use_center, bond_atom_sep=self.bond_atom_sep, no_ext=True)
        if (self.infile_dic is not None) and ('split' in self.infile_dic.keys()):
            name += '-split_{}.pt'.format(self.infile_dic['split'].split('.')[0])
        else:
            name += '.pt'
        return name

    def download(self):
        raise NotImplemented

    def process(self):
        phynet_data = np.load(osp.join(self.raw_dir, self.infile_dic['PhysNet']), allow_pickle=True)
        N = torch.as_tensor(phynet_data['N'])[:].long()
        R = torch.as_tensor(self._list2array(phynet_data['R'], 62, 'R'))[:, :, :].double()
        E = torch.as_tensor(phynet_data['E'])[:].double()
        Q = torch.as_tensor(phynet_data['Q'])[:].double()
        D = torch.as_tensor(phynet_data['D'])[:].double()
        Z = torch.as_tensor(self._list2array(phynet_data['Z'], 62, 'Z'))[:, :].long()
        num_mol = N.shape[0]

        if self.cal_efg:
            raise NotImplemented
        else:
            efgs_batch, EFG_R, EFG_Z, num_efg = None, None, None, None

        if self.bond_atom_sep:
            assert self.infile_dic['SDF'] is not None
            mol_list = torch.load(osp.join(self.raw_dir, self.infile_dic['SDF']))
        else:
            mol_list = None

        data_list = physnet_to_datalist(self, N, R, E, D, Q, Z, num_mol, mol_list, efgs_batch, EFG_R, EFG_Z, num_efg)
        print('collating...')
        data, slices = self.collate(data_list)
        print('saving...')

        if 'split' in self.infile_dic:
            split = torch.load(osp.join(self.processed_dir, self.infile_dic['split']))
        else:
            split = None
        torch.save((data, slices, split), self.processed_paths[0])

    @staticmethod
    def _list2array(source, max_num, pad_type):
        """
        convert list to numpy.array, implement padding
        :param source:
        :return:
        """
        if not isinstance(source[0], list):
            return source
        else:
            for i, data in enumerate(source):
                n_pad = max_num - len(data)
                item = [[0., 0., 0.]] if pad_type in ['R'] else [0]
                data.extend(item * n_pad)
            return np.asarray(source.tolist())


if __name__ == '__main__':
    _data = PhysDimeIMDataset(root='data', processed_prefix='CSD20_MMFF', pre_transform=my_pre_transform, record_long_range=True,
                              infile_dic={'PhysNet': osp.join("CSD20", "CSD20_PhysNet_MMFF.npz"), 'SDF': osp.join("CSD20", "CSD20_cry_min_MMFF.pt")})
    print('finished')
