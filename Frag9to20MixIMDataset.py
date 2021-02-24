import os
import os.path as osp
import time
import pandas as pd

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from DataPrepareUtils import name_extender, my_pre_transform, subtract_ref, sol_keys
from Frag20IMDataset import Frag20IMDataset

'''
Frag20 mixed data distribution presets
'''
uniform_split = {
    'name': 'uniform',
    'train': (8250, 8250, 8250, 8250, 8250, 8250, 8250, 8250, 8250, 8250, 8250, 8250),
    'valid': (83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83),
    'test': (5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, -1)
}

small_split = {
    'name': 'small',
    'train': (49500, 49500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'valid': (500, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'test': (5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, -1)
}

large_split = {
    'name': 'large',
    'train': (0, 0, 0, 0, 0, 0, 0, 0, 35000, 32000, 22500, 9500),
    'valid': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 500),
    'test': (5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, -1, -1, -1, -1)
}


class Frag9to20MixIMDataset(InMemoryDataset):
    def __init__(self, root, split_settings, transform=None, training_option='train', jianing_split=False,
                 all_data=False,
                 pre_transform=None, boundary_factor=100., type_3_body='B', extended_bond=False,
                 edge_version='cutoff', cutoff=10., cal_3body_term=True, sort_edge=True, use_center=True,
                 bond_atom_sep=True, cal_efg=False, record_long_range=False, frag20n9=False,
                 geometry='QM', add_sol=False, use_ref=True):
        """
        This in-memory dataset contains a mixture of Frag9 and Frag20
        :param root:
        :param transform:
        :param training_option: train | test | testDummy
        :param pre_transform:
        :param boundary_factor:
        :param edge_version:
        :param cutoff:
        :param cal_3body_term:
        :param sort_edge:
        :param use_center:
        :param bond_atom_sep:
        :param cal_efg:
        :param frag20n9: deprecated_code, always set to False.
        """
        self.use_ref = use_ref
        self.add_sol = add_sol
        self.geometry = geometry
        self.all_data = all_data
        self.jianing_split = jianing_split
        self.frag20n9 = frag20n9
        self.extended_bond = extended_bond
        self.training_option = training_option
        self.split_settings = split_settings
        self.type_3_body = type_3_body
        self.record_long_range = record_long_range

        self.bond_atom_sep = bond_atom_sep
        self.use_center = use_center
        self.sort_edge = sort_edge
        self.cutoff = cutoff
        self.cal_3body_term = cal_3body_term
        self.edge_version = edge_version
        self.boundary_factor = boundary_factor
        self.pre_transform = pre_transform
        self.transform = transform
        self.cal_efg = cal_efg

        super().__init__(root, transform, pre_transform)
        processed_file_id = ['train', 'testDummy', 'test'].index(self.training_option)
        if self.frag20n9 or self.jianing_split or self.all_data:
            self.data, self.slices = torch.load(self.processed_paths[processed_file_id])
            # if self.all_data:
            # split_selected_combined = np.load(os.path.join(self.raw_dir, 'split.npz'))
            # frag_split = {}
            # n_mol_before = 0
            # train_cum = []
            # test_cum = []
            # for i in range(20, 8, -1):
            #     frag_split[i] = np.load(os.path.join(self.raw_dir, 'fragment{}_split.npz'.format(i)))
            #     index_range = torch.arange(n_mol_before,
            #                                n_mol_before + len(frag_split[i]['train']) + len(frag_split[i]['test']))
            #     n_mol_before = n_mol_before + len(frag_split[i]['train']) + len(frag_split[i]['test'])
            #     train_cum.append(index_range[frag_split[i]['train']])
            #     test_cum.append(index_range[frag_split[i]['test']])
            # train_valid_index = torch.cat(train_cum, dim=0)
            # test_index = torch.cat(test_cum, dim=0)
            # self.val_index = train_valid_index[split_selected_combined['validation']]
            # self.train_index = torch.cat([train_valid_index[split_selected_combined['train']],
            #                               train_valid_index[split_selected_combined['test']],
            #                               test_index])
            # self.test_index = torch.arange(10)  # dummy test index
            # if self.jianing_split or self.all_data:
            split_file = np.load(self.raw_paths[0])
            if self.training_option in ['train', 'testDummy']:
                self.train_index = torch.arange(len(split_file['train']))
                self.val_index = torch.arange(len(split_file['train']),
                                              len(split_file['train']) + len(split_file['validation']))
                self.test_index = torch.arange(len(split_file['train']) + len(split_file['validation']),
                                               len(split_file['train']) + len(split_file['validation']) + len(
                                                   split_file['test']))
                if self.all_data:
                    self.train_index = torch.cat([self.train_index, self.test_index])
            elif self.training_option == 'test':
                self.test_index = torch.arange(len(self))

            if self.add_sol:
                sol_split_file = osp.join(self.processed_dir, "frag20_sol_split.pt")
                if not osp.exists(sol_split_file):
                    perm_tensor = torch.randperm(len(self))
                    _split = {"train_index": perm_tensor[:-41000],
                              "val_index": perm_tensor[-41000: -40000],
                              "test_index": perm_tensor[-40000:]}
                    torch.save(_split, sol_split_file)
                else:
                    _split = torch.load(sol_split_file)
                for key in _split:
                    setattr(self, key, _split[key])
        else:
            self.data, self.slices, meta_data = torch.load(self.processed_paths[processed_file_id])
            self.frag20_size = meta_data['frag20_size'],

            self.train_index_perm, self.val_index_perm, self.test_index_perm = meta_data['train_index_perm'], \
                                                                               meta_data['valid_index_perm'], \
                                                                               meta_data['test_index_perm']

            self.train_size = 0
            self.val_size = 0
            self.test_sXize = 0
            self.index_init()
            self.data_size = self.train_size + self.val_size + self.test_size

            self.train_index = torch.arange(self.train_size)
            self.val_index = torch.arange(self.train_size, self.train_size + self.val_size)
            # since we separate test from train and valid, test index was set to 0-test size
            self.test_index = torch.arange(0, self.test_size)

            self.val_index_separate = self.index_separate(self.train_size, self.val_index_perm)
            self.test_index_separate = self.index_separate(0, self.test_index_perm)

    def index_init(self):
        total_train_size = 0
        for train_index in self.train_index_perm.values():
            total_train_size = total_train_size + len(train_index)
        self.train_size = total_train_size

        total_valid_size = 0
        for valid_index in self.val_index_perm.values():
            total_valid_size = total_valid_size + len(valid_index)
        self.val_size = total_valid_size

        total_test_size = 0
        for test_index in self.test_index_perm.values():
            total_test_size = total_test_size + len(test_index)
        self.test_size = total_test_size

    @staticmethod
    def index_separate(previous_size, index_perm):
        result = {}
        for n_heavy_atom in range(9, 21):
            result[n_heavy_atom] = torch.arange(previous_size, previous_size + len(index_perm[n_heavy_atom]))
            previous_size = previous_size + len(index_perm[n_heavy_atom])
        return result

    @property
    def raw_file_names(self):
        # NOTE: in order to skip download, we added a dummy raw file
        # this file is only required when self.jianing_split == True
        names = ['split.npz']
        return names

    @property
    def processed_file_names(self):
        if self.frag20n9:
            return ['frag20n9mix-strict-Bmsg-cutoff-10.00-sorted-defined_edge-lr-sample-(99000, 9000)-(1000, '
                    '700)-(500, 500).pt']
        elif self.all_data:
            print('WARNING: using all frag20 data')
            name = 'frag20reducedAll'
            if self.add_sol:
                name += "Sol"
                if self.use_ref:
                    name += "Ref"
            name = name_extender(name, cal_3body_term=self.cal_3body_term, edge_version=self.edge_version,
                                 cutoff=self.cutoff, boundary_factor=self.boundary_factor,
                                 extended_bond=self.extended_bond, geometry=self.geometry,
                                 use_center=self.use_center, bond_atom_sep=self.bond_atom_sep,
                                 record_long_range=self.record_long_range, type_3_body=self.type_3_body)
            return [name]
        else:
            names = []
            for training_option in ['train', 'testDummy', 'test']:
                if self.jianing_split:
                    name = 'frag20jianing'
                else:
                    name = 'frag9to20mix'
                if self.add_sol:
                    name += "Sol"
                name = name_extender(name, cal_3body_term=self.cal_3body_term, edge_version=self.edge_version,
                                     cutoff=self.cutoff, boundary_factor=self.boundary_factor,
                                     extended_bond=self.extended_bond, geometry=self.geometry,
                                     use_center=self.use_center, bond_atom_sep=self.bond_atom_sep,
                                     record_long_range=self.record_long_range, type_3_body=self.type_3_body)
                name = name[:-3] + '(sample-{})'.format(self.split_settings['name'])
                name = name + '-{}.pt'.format(training_option)
                names.append(name)
            names.append('sample-{}.pt'.format(self.split_settings['name']))
            return names

    def download(self):
        raise ValueError('NotImplemented')

    def process(self):

        if self.all_data:
            if not self.add_sol:
                data_list = []
                jianing_train = Frag9to20MixIMDataset(root=self.root, split_settings=self.split_settings,
                                                      transform=self.transform, training_option='train',
                                                      jianing_split=True,
                                                      boundary_factor=self.boundary_factor,
                                                      type_3_body=self.type_3_body,
                                                      extended_bond=self.extended_bond, geometry=self.geometry,
                                                      edge_version=self.edge_version, cutoff=self.cutoff,
                                                      cal_3body_term=self.cal_3body_term,
                                                      sort_edge=self.sort_edge, use_center=self.use_center,
                                                      bond_atom_sep=self.bond_atom_sep, cal_efg=self.cal_efg,
                                                      record_long_range=self.record_long_range, add_sol=self.add_sol)
                jianing_test = Frag9to20MixIMDataset(root=self.root, split_settings=self.split_settings,
                                                     transform=self.transform, training_option='testDummy',
                                                     boundary_factor=self.boundary_factor, type_3_body=self.type_3_body,
                                                     extended_bond=self.extended_bond, jianing_split=True,
                                                     edge_version=self.edge_version, cutoff=self.cutoff,
                                                     cal_3body_term=self.cal_3body_term, geometry=self.geometry,
                                                     sort_edge=self.sort_edge, use_center=self.use_center,
                                                     bond_atom_sep=self.bond_atom_sep, cal_efg=self.cal_efg,
                                                     record_long_range=self.record_long_range, add_sol=self.add_sol)
                for i in range(len(jianing_train)):
                    data_list.append(jianing_train[i])
                del jianing_train
                for i in range(len(jianing_test)):
                    data_list.append(jianing_test[i])
                del jianing_test

            else:
                jianing_to_dongdong_map = []
                data_list = []
                frag20_info = {i: pd.read_csv("data/raw/Frag20ValidCheck/Frag20_{}_1D_infor.csv".format(i)) for i in
                               range(9, 21)}
                frag20_sol = pd.read_csv("data/raw/frag20/frag20_solvation.csv")
                for n_heavy in tqdm(range(9, 21)):
                    data = Frag20IMDataset(root=self.root, transform=self.transform,
                                           pre_transform=self.pre_transform, geometry=self.geometry,
                                           boundary_factor=self.boundary_factor, type_3_body=self.type_3_body,
                                           edge_version=self.edge_version, cutoff=self.cutoff,
                                           extended_bond=self.extended_bond,
                                           cal_3body_term=self.cal_3body_term, sort_edge=self.sort_edge,
                                           use_center=self.use_center,
                                           record_long_range=self.record_long_range,
                                           bond_atom_sep=self.bond_atom_sep, cal_efg=self.cal_efg,
                                           n_heavy_atom=n_heavy,
                                           add_sol=False)
                    info_csv = frag20_info[n_heavy]
                    col_name = "QM_InChI" if n_heavy > 9 else "opt_Inchi"
                    source_file = "less10" if n_heavy == 9 else str(n_heavy)
                    frag20_sol_i = frag20_sol.loc[frag20_sol["SourceFile"].values.astype(np.str) == source_file]
                    frag20_sol_inchi_list = frag20_sol_i["InChI"].values.tolist()
                    # print(frag20_sol_inchi_list)
                    # exit()
                    for i in tqdm(range(len(data))):
                        InChI = info_csv[col_name][i]
                        try:
                            this_sol_data = frag20_sol_i.iloc[frag20_sol_inchi_list.index(InChI)]
                            # print(this_sol_data)
                        except ValueError:
                            this_sol_data = None
                        if this_sol_data is not None:
                            jianing_to_dongdong_map.append(1)
                            _tmp_data = data[i]
                            for key in sol_keys:
                                _tmp_data.__setattr__(key, torch.as_tensor(this_sol_data[key]).view(-1))
                            _tmp_data.InChI = np.asarray([InChI], dtype=np.str)
                            data_list.append(_tmp_data)
                        else:
                            jianing_to_dongdong_map.append(0)
                torch.save(torch.as_tensor(jianing_to_dongdong_map),
                           "jianing_to_dongdong_map_{}_merge.pt".format(self.geometry))

            print('collating...')
            data, slices = self.collate(data_list)
            print('saving...')
            torch.save((data, slices), self.processed_paths[0])
            return

        frag20_data = {}
        for i in range(9, 21):
            frag20_data[i] = Frag20IMDataset(root=self.root, transform=self.transform,
                                             pre_transform=self.pre_transform, geometry=self.geometry,
                                             boundary_factor=self.boundary_factor, type_3_body=self.type_3_body,
                                             edge_version=self.edge_version, cutoff=self.cutoff,
                                             extended_bond=self.extended_bond,
                                             cal_3body_term=self.cal_3body_term, sort_edge=self.sort_edge,
                                             use_center=self.use_center, record_long_range=self.record_long_range,
                                             bond_atom_sep=self.bond_atom_sep, cal_efg=self.cal_efg, n_heavy_atom=i,
                                             add_sol=self.add_sol)

        if self.jianing_split:
            split = {}
            test = {}
            for n_heavy_atom in range(9, 21):
                split_file = os.path.join(self.raw_dir, 'fragment{}_split.npz'.format(n_heavy_atom))
                split[n_heavy_atom] = np.load(split_file)['train']
                test[n_heavy_atom] = np.load(split_file)['test']
            '''
            Do some math trick to select train and test
            '''
            index = torch.cat([torch.as_tensor(split[i]) for i in range(20, 8, -1)])
            n_heavy_atom_tensor = torch.cat([torch.zeros(len(split[i])).long().fill_(i) for i in range(20, 8, -1)])
            index = torch.stack([index, n_heavy_atom_tensor])
            split_file = np.load(os.path.join(self.raw_dir, 'split.npz'))
            train_split, valid_split, test_split = split_file['train'], split_file['validation'], split_file['test']
            train_valid_original = index[:, torch.cat([torch.as_tensor(train_split), torch.as_tensor(valid_split)])]
            test_original = index[:, torch.as_tensor(test_split)]
            train_data_list = []
            test_dummy_data_list = []
            test_data_list = []

            if not os.path.exists(os.path.join(self.processed_dir, self.processed_file_names[0])):
                print('processing training and validation data')
                for i in range(train_valid_original.shape[1]):
                    n_heavy_atom = train_valid_original[1][i].item()
                    inner_id = train_valid_original[0][i].item()
                    train_data_list.append(frag20_data[n_heavy_atom][inner_id])
                print('collating...')
                data, slices = self.collate(train_data_list)
                print('saving...')
                torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[0]))
                del train_data_list, data, slices

            if not os.path.exists(os.path.join(self.processed_dir, self.processed_file_names[1])):
                print('processing test dummy data')
                for i in range(test_original.shape[1]):
                    n_heavy_atom = test_original[1][i].item()
                    inner_id = test_original[0][i].item()
                    test_dummy_data_list.append(frag20_data[n_heavy_atom][inner_id])
                print('collating...')
                data, slices = self.collate(test_dummy_data_list)
                print('saving...')
                torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[1]))
                del test_dummy_data_list, data, slices

            if not os.path.exists(os.path.join(self.processed_dir, self.processed_file_names[2])):
                print('processing (true) test data')
                for n_heavy_atom in range(20, 8, -1):
                    for test_index in test[n_heavy_atom]:
                        test_data_list.append(frag20_data[n_heavy_atom][int(test_index)])
                print('collating...')
                data, slices = self.collate(test_data_list)
                print('saving...')
                torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[2]))
        else:
            frag20_size = {
                i: len(frag20_data[i]) for i in range(9, 21)
            }

            if not os.path.exists(self.processed_paths[1]):
                print('sample file not found, random generating one')
                train_split = self.split_settings['train']
                val_split = self.split_settings['valid']
                test_split = self.split_settings['test']
                train_index, val_index, test_index = {}, {}, {}
                for num, (train_size, val_size, test_size) in enumerate(zip(train_split, val_split, test_split)):
                    n_heavy_atom = num + 9
                    perm_tensor = torch.randperm(frag20_size[n_heavy_atom])
                    train_index[n_heavy_atom] = perm_tensor[:train_size]
                    val_index[n_heavy_atom] = perm_tensor[train_size:train_size + val_size]
                    if test_size < 0:
                        test_index[n_heavy_atom] = perm_tensor[train_size + val_size:]
                    else:
                        test_index[n_heavy_atom] = perm_tensor[train_size + val_size: train_size + val_size + test_size]
                    torch.save((train_index, val_index, test_index), self.processed_paths[1])
            else:
                print('loading existing sample file...')
                train_index, val_index, test_index = torch.load(self.processed_paths[1])

            data_list = []
            if self.training_option == 'train':
                for n_heavy_atom in range(9, 21):
                    data_list.extend(frag20_data[n_heavy_atom][train_index[n_heavy_atom]])
                for n_heavy_atom in range(9, 21):
                    data_list.extend(frag20_data[n_heavy_atom][val_index[n_heavy_atom]])
            else:
                for n_heavy_atom in range(9, 21):
                    data_list.extend(frag20_data[n_heavy_atom][test_index[n_heavy_atom]])

            meta_data = {
                'frag20_size': frag20_size,
                'train_index_perm': train_index,
                'valid_index_perm': val_index,
                'test_index_perm': test_index
            }
            print('collating...')
            data, slices = self.collate(data_list)
            print('saving...')
            torch.save((data, slices, meta_data), self.processed_paths[0])


if __name__ == '__main__':
    _dataset = Frag9to20MixIMDataset(root='../dataProviders/data', split_settings=uniform_split, record_long_range=True,
                                     pre_transform=my_pre_transform, bond_atom_sep=True, cal_3body_term=True,
                                     training_option='train', all_data=True, jianing_split=True, geometry='QM',
                                     add_sol=True, use_ref=False)
    subtract_ref(_dataset, "data/processed/frag20reducedAllSolRef-Bmsg-cutoff-10.00-sorted-defined_edge-lr-QM.pt")
    print(len(_dataset))
