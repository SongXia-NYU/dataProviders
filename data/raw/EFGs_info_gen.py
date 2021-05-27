import torch
import numpy as np
from EFGs.three_level_frag import mol2frag
from rdkit.Chem import PeriodicTable, GetPeriodicTable
import logging
import time


def _test_consistency(test_idx):
    tmp = mol2frag(qm9_mol[test_idx], TreatHs='include', vocabulary=qm9_vocab, returnidx=True, toEnd=True,
                   extra_included=True,
                   isomericSmiles=True)
    tmp1 = qm9_data['Z'][test_idx]
    return


if __name__ == '__main__':
    """
    Generate EFGs related data for QM9
    First, mol files is loaded into variable 'qm9_mol'
    Then, loop through every mol to calculate the EFGs of each mol. 
        Batch related information is stored in variable 'efgs_batch', 
        the mass center of each efg is stored in variable 'efgs_mass_center'.
    Finally, save every data into a dictionary.
    """
    table = GetPeriodicTable()
    zToWeightMap = {}
    for Z in range(1, 95):
        zToWeightMap[Z] = PeriodicTable.GetAtomicWeight(table, Z)

    # Logger setup
    logging.basicConfig(filename='LOG_EFGs_batch_gen.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    qm9_data = np.load('qm9_removeproblem.npz')
    qm9_data_shape = qm9_data['R'].shape
    num_mols = qm9_data_shape[0]
    max_atoms = qm9_data_shape[1]

    # Data to be recorded
    # efgs_batch:        is used to generate batch data to merge atoms in the same efg
    # efgs_mass_center:  records data of the mass center of that efg
    # efgs_to_mol_mask:  is used to generate batch data to merge efgs in the same molecule
    # num_efgs:          total number of efgs
    # efg_types:           The exact type of the EFG, recorded by id
    # max_efg_types:     Max types of EFGs
    efgs_batch = torch.LongTensor(num_mols, max_atoms).fill_(-1)
    efgs_mass_center = np.zeros((num_mols, max_atoms, 3))
    efgs_to_mol_mask = torch.BoolTensor(num_mols, max_atoms).fill_(False)
    num_efgs = torch.LongTensor(num_mols).fill_(0)
    efg_types = torch.LongTensor(num_mols, max_atoms).fill_(-1)

    qm9_mol = torch.load('QM9SDFs_removeproblem.pt')
    qm9_vocab = torch.load('QM9_vocab_add3D_0802_cutoff0.7_EFGs_frequency_addatom_addnone.pt')
    max_efg_types = len(qm9_vocab)

    efg_name_to_id = {}
    for num, efg_name in enumerate(qm9_vocab):
        efg_name_to_id[efg_name] = num

    Z_data = qm9_data['Z']
    R_data = qm9_data['R']

    for i, mol in enumerate(qm9_mol):
        efg_id = 0
        fg_names, efg_names, fg_group, efg_group = mol2frag(mol, TreatHs='include', vocabulary=qm9_vocab,
                                                            returnidx=True,
                                                            toEnd=True,
                                                            extra_included=True, isomericSmiles=True)
        for efg_name, one_efg_group in zip(fg_names + efg_names, fg_group + efg_group):
            '''
            Loop through every efg in one molecule
            '''

            # init weighted coordinates and total mass to calculate mass center
            _tmp_mass_center = np.zeros((3,))
            _tmp_total_mass = np.zeros((1,))

            for atom_id in one_efg_group:
                '''
                Loop through every atom in one efg
                record efg id and calculate mass center
                '''
                # t0 = time.time()
                efgs_batch[i][atom_id] = efg_id
                # print('step 1: ', time.time()-t0)
                mass = zToWeightMap[Z_data[i, atom_id]]
                # print('step 2: ', time.time() - t0)
                R_atom = R_data[i, atom_id, :]
                # print('step 3: ', time.time() - t0)
                _tmp_mass_center += R_atom * mass
                # print('step 4: ', time.time() - t0)
                _tmp_total_mass += mass
                # print('step 5: ', time.time() - t0)

            # record calculated mass center
            efgs_mass_center[i, efg_id, :] = _tmp_mass_center / _tmp_total_mass

            # record mask information
            efgs_to_mol_mask[i, efg_id] = True

            # record the id of this efg
            efg_types[i, efg_id] = efg_name_to_id[efg_name]
            efg_id += 1

        # At the end of loop through each EFGs, efg_id becomes the total number of EFGs in this molecule
        num_efgs[i] = efg_id

        if (i + 1) % 500 == 0:
            # logging.info('Process: {} %'.format(float(i * 100) / num_mols))
            print('Process: {} %'.format(float(i * 100) / num_mols))

    torch.save({'efgs_batch': efgs_batch,
                'efgs_mass_center': efgs_mass_center,
                'efgs_to_mol_mask': efgs_to_mol_mask,
                'num_efgs': num_efgs,
                'efg_types': efg_types,
                'max_efg_types': max_efg_types}, 'EFGs_QM9.pt')

    print('Boo! Finished!')
