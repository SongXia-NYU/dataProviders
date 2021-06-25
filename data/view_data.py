import torch
import numpy as np
import matplotlib.pyplot as plt

from dataProviders.qm9InMemoryDataset import Qm9InMemoryDataset


def _dist_from_R_edge(R, edge):
    point1 = R[edge[0, :], :]
    point2 = R[edge[1, :], :]
    dist = torch.sum((point1-point2)**2, dim=-1, keepdim=True)
    dist = torch.sqrt(dist)
    return dist.view(-1)


if __name__ == '__main__':
    qm9Data = Qm9InMemoryDataset(root='.')
    num_mol = qm9Data.data.E.shape[0]
    atom_edge_dist = torch.cat(
        [_dist_from_R_edge(qm9Data[i].R, qm9Data[i].atom_edge_index) for i in range(num_mol)])
    plt.hist(atom_edge_dist, bins=500)
    plt.xlabel('distance/A')
    plt.ylabel('count')
    plt.title('Distance distribution of atom-wise bond')
    plt.show()
    efg_edge_dist = torch.cat(
        [_dist_from_R_edge(qm9Data[i].EFG_R, qm9Data[i].EFG_edge_index) for i in range(num_mol)])
    plt.hist(efg_edge_dist, bins=500)
    plt.xlabel('distance/A')
    plt.ylabel('count')
    plt.title('Distance distribution of EFG-wise bond')
    plt.show()
    print('Finished.')
