import time
from typing import List

import numpy as np
import pandas as pd
import torch
import os.path as osp

import torch_geometric
from scipy.spatial import Voronoi
from torch_geometric.data import Data
from ase.units import Hartree, eV

from DummyIMDataset import DummyIMDataset
from utils.utils_functions import cal_edge
from tqdm import tqdm

hartree2ev = Hartree / eV


def scale_R(R):
    abs_min = torch.abs(R).min()
    while abs_min < 1e-3:
        R = R - 1
        abs_min = torch.abs(R).min()
    return R


def cal_msg_edge_index(edge_index):
    msg_id_1 = torch.arange(edge_index.shape[-1]).repeat(edge_index.shape[-1], 1)
    msg_id_0 = msg_id_1.t()
    source_atom = edge_index[0, :].repeat(edge_index.shape[-1], 1)
    target_atom = edge_index[1, :].view(-1, 1)
    msg_map = (source_atom == target_atom)
    result = torch.cat([msg_id_0[msg_map].view(1, -1), msg_id_1[msg_map].view(1, -1)], dim=0)
    return result


def voronoi_edge_index(R, boundary_factor, use_center):
    """
    Calculate Voronoi Diagram
    :param R: shape[-1, 3], the location of input points
    :param boundary_factor: Manually setup a boundary for those points to avoid potential error, value of [1.1, inf]
    :param use_center: If true, the boundary will be centered on center of points; otherwise, boundary will be centered
    on [0., 0., 0.]
    :return: calculated edge index
    """
    R = scale_R(R)

    R_center = R.mean(dim=0) if use_center else torch.DoubleTensor([0, 0, 0])

    # maximum relative coordinate
    max_coordinate = torch.abs(R - R_center).max()
    boundary = max_coordinate * boundary_factor
    appended_R = torch.zeros(8, 3).double().fill_(boundary)
    idx = 0
    for x_sign in [-1, 1]:
        for y_sign in [-1, 1]:
            for z_sign in [-1, 1]:
                appended_R[idx] *= torch.DoubleTensor([x_sign, y_sign, z_sign])
                idx += 1
    num_atoms = R.shape[0]

    appended_R = appended_R + R_center
    diagram = Voronoi(torch.cat([R, appended_R], dim=0), qhull_options="Qbb Qc Qz")
    edge_one_way = diagram.ridge_points
    edge_index_all = torch.LongTensor(np.concatenate([edge_one_way, edge_one_way[:, [1, 0]]], axis=0)).t()
    mask0 = edge_index_all[0, :] < num_atoms
    mask1 = edge_index_all[1, :] < num_atoms
    mask = mask0 & mask1
    edge_index = edge_index_all[:, mask]

    return edge_index


def sort_edge(edge_index):
    """
    sort the target of edge to be sequential, which may increase computational efficiency later on when training
    :param edge_index:
    :return:
    """
    arg_sort = torch.argsort(edge_index[1, :])
    return edge_index[:, arg_sort]


def mol_to_edge_index(mol):
    """
    Calculate edge_index(bonding edge) from rdkit.mol
    :param mol:
    :return:
    """
    bonds = mol.GetBonds()
    num_bonds = len(bonds)
    _edge_index = torch.zeros(2, num_bonds).long()
    for bond_id, bond in enumerate(bonds):
        _edge_index[0, bond_id] = bond.GetBeginAtomIdx()
        _edge_index[1, bond_id] = bond.GetEndAtomIdx()
    _edge_index_inv = _edge_index[[1, 0], :]
    _edge_index = torch.cat([_edge_index, _edge_index_inv], dim=-1)
    return _edge_index


def remove_bonding_edge(all_edge_index, bond_edge_index):
    """
    Remove bonding index from atom_edge_index to avoid double counting
    :param all_edge_index:
    :param bond_edge_index:
    :return:
    """
    mask = torch.zeros(all_edge_index.shape[-1]).bool().fill_(False)
    len_bonding = bond_edge_index.shape[-1]
    for i in range(len_bonding):
        same_atom = (all_edge_index == bond_edge_index[:, i].view(-1, 1))
        mask += (same_atom[0] & same_atom[1])
    remain_mask = ~ mask
    return all_edge_index[:, remain_mask]


def extend_bond(edge_index):
    """
    extend bond edge to a next degree, i.e. consider all 1,3 interaction as bond
    :param edge_index:
    :return:
    """
    n_edge = edge_index.size(-1)
    source = edge_index[0]
    target = edge_index[1]

    # expand into a n*n matrix
    source_expand = source.repeat(n_edge, 1)
    target_t = target.view(-1, 1)

    mask = (source_expand == target_t)
    target_index_mapper = edge_index[1].repeat(n_edge, 1)
    source_index_mapper = edge_index[0].repeat(n_edge, 1).t()

    source_index = source_index_mapper[mask]
    target_index = target_index_mapper[mask]

    extended_bond = torch.cat([source_index.view(1, -1), target_index.view(1, -1)], dim=0)
    # remove self to self interaction
    extended_bond = extended_bond[:, source_index != target_index]
    extended_bond = remove_bonding_edge(extended_bond, edge_index)
    result = torch.cat([edge_index, extended_bond], dim=-1)
    return result


def my_pre_transform(data, edge_version, do_sort_edge, cal_efg, cutoff, boundary_factor, use_center, mol,
                     cal_3body_term, bond_atom_sep, record_long_range, type_3_body='B', extended_bond=False):
    """
    edge calculation
    atom_edge_index is non-bonding edge index when bond_atom_sep=True; Otherwise, it is bonding and non-bonding together
    """
    edge_index = torch.zeros(2, 0).long()
    dist, full_edge, _, _ = cal_edge(data.R, [data.N], [0], edge_index, cal_coulomb=True)
    dist = dist.cpu()
    full_edge = full_edge.cpu()

    if edge_version == 'cutoff':
        data.BN_edge_index = full_edge[:, (dist < cutoff).view(-1)]
    else:
        data.BN_edge_index = voronoi_edge_index(data.R, boundary_factor, use_center=use_center)

    if record_long_range:
        data.L_edge_index = remove_bonding_edge(full_edge, data.BN_edge_index)

    '''
    sort edge index
    '''
    if do_sort_edge:
        data.BN_edge_index = sort_edge(data.BN_edge_index)

    '''
    EFGs edge calculation
    '''
    if cal_efg:
        if edge_version == 'cutoff':
            dist, full_edge, _, _ = cal_edge(data.EFG_R, [data.EFG_N], [0], edge_index, cal_coulomb=True)
            data.EFG_edge_index = full_edge[:, (dist < cutoff).view(-1)].cpu()
        else:
            data.EFG_edge_index = voronoi_edge_index(data.EFG_R, boundary_factor, use_center=use_center)

        data.num_efg_edges = torch.LongTensor([data.EFG_edge_index.shape[-1]]).view(-1)

    if bond_atom_sep:
        '''
        Calculate bonding edges and remove those non-bonding edges which overlap with bonding edge
        '''
        if mol is None:
            print('rdkit mol file not given for molecule: {}, cannot calculate bonding edge, skipping this'.format(
                data.Z))
            return None
        B_edge_index = mol_to_edge_index(mol)
        if B_edge_index.max() + 1 > data.N:
            raise ValueError('problematic mol file: {}'.format(mol))
        if extended_bond:
            B_edge_index = extend_bond(B_edge_index)
        if do_sort_edge:
            B_edge_index = sort_edge(B_edge_index)
        data.B_edge_index = B_edge_index
        data.N_edge_index = remove_bonding_edge(data.BN_edge_index, B_edge_index)
        _edge_list = []
        for bond_type in type_3_body:
            _edge_list.append(getattr(data, bond_type + "_edge_index"))
        _edge_index = torch.cat(_edge_list, dim=-1)
    else:
        _edge_index = data.BN_edge_index

    '''
    Calculate 3-atom term(Angle info)
    It ls essentially an "edge" of edge
    '''
    if cal_3body_term:

        atom_msg_edge_index = cal_msg_edge_index(_edge_index)
        if do_sort_edge:
            atom_msg_edge_index = sort_edge(atom_msg_edge_index)

        setattr(data, type_3_body + '_msg_edge_index', atom_msg_edge_index)

        setattr(data, 'num_' + type_3_body + '_msg_edge', torch.zeros(1).long() + atom_msg_edge_index.shape[-1])

    for bond_type in ['B', 'N', 'L', 'BN']:
        _edge_index = getattr(data, bond_type + '_edge_index', False)
        if _edge_index is not False:
            setattr(data, 'num_' + bond_type + '_edge', torch.zeros(1).long() + _edge_index.shape[-1])

    return data


def name_extender(name, cal_3body_term=None, edge_version=None, cutoff=None, boundary_factor=None, use_center=None,
                  bond_atom_sep=None, record_long_range=False, type_3_body='B', extended_bond=False, no_ext=False,
                  geometry='QM'):
    if extended_bond:
        type_3_body = type_3_body + 'Ext'
    name += '-' + type_3_body
    if cal_3body_term:
        name += 'msg'

    if edge_version == 'cutoff':
        if cutoff is None:
            print('cutoff canot be None when edge version == cutoff, exiting...')
            exit(-1)
        name += '-cutoff-{:.2f}'.format(cutoff)
    elif edge_version == 'voronoi':
        name += '-box-{:.2f}'.format(boundary_factor)
        if use_center:
            name += '-centered'
    else:
        raise ValueError('Cannot recognize edge version(neither cutoff or voronoi), got {}'.format(edge_version))

    if sort_edge:
        name += '-sorted'

    if bond_atom_sep:
        name += '-defined_edge'

    if record_long_range:
        name += '-lr'

    name += '-{}'.format(geometry)

    if not no_ext:
        name += '.pt'
    return name


sol_keys = ["gasEnergy", "watEnergy", "octEnergy", "CalcSol", "CalcOct", "calcLogP"]


def physnet_to_datalist(self, N, R, E, D, Q, Z, num_mol, mols, efgs_batch, EFG_R, EFG_Z, num_efg, sol_data=None):
    """
    load data from PhysNet structure to InMemoryDataset structure (more compact)
    :return:
    """
    from rdkit.Chem.inchi import MolToInchi

    data_array = np.empty(num_mol, dtype=Data)
    t0 = time.time()
    Z_0 = Z[0, :]
    n_heavy = len(Z_0) - (Z_0 == 0).sum() - (Z_0 == 1).sum()

    jianing_to_dongdong_map = []

    for i in tqdm(range(num_mol)):
        if self.bond_atom_sep:
            mol = mols[i]
        else:
            mol = None
        # atomic infos
        _tmp_Data = Data()

        num_atoms = N[i]
        _tmp_Data.N = num_atoms.view(-1)
        _tmp_Data.R = R[i, :N[i]].view(-1, 3)
        _tmp_Data.E = E[i].view(-1)
        _tmp_Data.D = D[i].view(-1, 3)
        _tmp_Data.Q = Q[i].view(-1)
        _tmp_Data.Z = Z[i, :N[i]].view(-1)

        if self.cal_efg:
            _tmp_Data.atom_to_EFG_batch = efgs_batch[i, :N[i]].view(-1)
            _tmp_Data.EFG_R = EFG_R[i, :num_efg[i]].view(-1, 3)
            _tmp_Data.EFG_Z = EFG_Z[i, :num_efg[i]].view(-1)
            _tmp_Data.EFG_N = num_efg[i].view(-1)

        if sol_data is not None:
            # find molecule from solvation csv file based on InChI, if found, add it
            this_sol_data = sol_data.loc[sol_data["InChI"] == MolToInchi(mol)]
            if this_sol_data.shape[0] == 1:
                for key in sol_keys:
                    _tmp_Data.__setattr__(key, torch.as_tensor(this_sol_data.iloc[0][key]).view(-1))
                jianing_to_dongdong_map.append(1)
            else:
                jianing_to_dongdong_map.append(0)
                continue

        _tmp_Data = self.pre_transform(data=_tmp_Data, edge_version=self.edge_version, do_sort_edge=self.sort_edge,
                                       cal_efg=self.cal_efg, cutoff=self.cutoff, extended_bond=self.extended_bond,
                                       boundary_factor=self.boundary_factor, type_3_body=self.type_3_body,
                                       use_center=self.use_center, mol=mol, cal_3body_term=self.cal_3body_term,
                                       bond_atom_sep=self.bond_atom_sep, record_long_range=self.record_long_range)

        data_array[i] = _tmp_Data

    if sol_data is not None:
        torch.save(torch.as_tensor(jianing_to_dongdong_map), "jianing_to_dongdong_map_{}.pt".format(n_heavy))

    data_list = [data_array[i] for i in range(num_mol) if data_array[i] is not None]

    return data_list


def remove_atom_from_dataset(atom_z, dataset, remove_split=('train', 'valid', 'test'), explicit_split=None,
                             return_mask=False):
    """
    remove a specific atom from dataset
    H: 1
    B: 5
    C: 6
    N: 7
    O: 8
    F: 9
    :return:
    new train, valid, test split
    """
    if explicit_split is not None:
        index_getter = {
            'train': explicit_split[0],
            'valid': explicit_split[1],
            'test': explicit_split[2]
        }
    else:
        index_getter = {
            'train': getattr(dataset, 'train_index', 'none'),
            'valid': getattr(dataset, 'val_index', 'none'),
            'test': getattr(dataset, 'test_index', 'none')
        }
    removed_index = {'train': None, 'valid': None, 'test': None}
    mask_dict = {'train': None, 'valid': None, 'test': None}
    for key in remove_split:
        if index_getter[key] == 'none':
            removed_index[key] = None
        else:
            index = torch.as_tensor(index_getter[key])
            mask = torch.zeros_like(index).bool().fill_(True)
            for num, i in enumerate(index):
                if torch.sum(dataset[i.item()].Z == atom_z) > 0:
                    mask[num] = False
            removed_index[key] = index[mask]
            mask_dict[key] = mask
    if return_mask:
        return mask_dict['train'], mask_dict['valid'], mask_dict['test']
    else:
        return removed_index['train'], removed_index['valid'], removed_index['test']


def subtract_ref(dataset, save_path, use_jianing_ref=True, data_root="./data"):
    """
    Subtracting reference energy, the result is in eV unit
    :param data_root:
    :param dataset:
    :param save_path:
    :param use_jianing_ref:
    :return:
    """
    if save_path is not None and osp.exists(save_path):
        raise ValueError("cannot overwrite existing file!!!")
    if use_jianing_ref:
        ref_data = np.load(osp.join(data_root, "raw/atomref.B3LYP_631Gd.10As.npz"))
        u0_ref = ref_data["atom_ref"][:, 1]
    else:
        ref_data = pd.read_csv(osp.join(data_root, "raw/atom_ref_gas.csv"))
        u0_ref = np.zeros(96, dtype=np.float)
        for i in range(ref_data.shape[0]):
            u0_ref[int(ref_data.iloc[i]["atom_num"])] = float(ref_data.iloc[i]["energy(eV)"])
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        total_ref = u0_ref[data.Z].sum()
        for prop in ["watEnergy", "octEnergy", "gasEnergy"]:
            energy = getattr(data, prop)
            energy *= hartree2ev
            energy -= total_ref
    if save_path is not None:
        torch.save((dataset.data, dataset.slices), save_path)


def concat_im_datasets(root: str, datasets: List[str], out_name: str):
    data_list = []
    for dataset in datasets:
        dummy_dataset = DummyIMDataset(root, dataset)
        for i in tqdm(range(len(dummy_dataset)), dataset):
            data_list.append(dummy_dataset[i])
    print("saving... it is recommended to have 32GB memory")
    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               osp.join(root, "processed", out_name))
