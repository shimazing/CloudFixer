import os
import glob
import random

import numpy as np
import torch


def save_xyz_file(path, positions, id_from=0,
        name='pointcloud', node_mask=None, n_nodes=1024):
    try:
        os.makedirs(path)
    except OSError:
        pass

    batch_size = positions.shape[0]

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [n_nodes]  * batch_size

    for batch_i in range(batch_size):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % atomsxmol[batch_i])
        for atom_i in range(n_nodes):
            atom = 'o'
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()


def load_xyz(file):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files


def visualize(path, max_num=25, wandb=None, postfix=''):
    files = load_xyz_files(path)[0:max_num]
    for file in files:
        positions = load_xyz(file)
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
        dists = dists[dists > 0]
        print("Average distance between atoms", dists.mean().item())

        if wandb is not None:
            path = file[:-4] + '.png'
            # Log image(s)
            obj3d = wandb.Object3D({
                "type": "lidar/beta",
                "points": positions.cpu().numpy().reshape(-1, 3),
                "boxes": np.array(
                    [
                        {
                            "corners": (np.array([
                                [-1, -1, -1],
                                [-1, 1, -1],
                                [-1, -1, 1],
                                [1, -1, -1],
                                [1, 1, -1],
                                [-1, 1, 1],
                                [1, -1, 1],
                                [1, 1, 1]
                                ])*3).tolist(),
                            "label": "Box",
                            "color": [123, 321, 111], # ???
                        }
                    ]
                ),
                })
            wandb.log({'3d' + postfix: obj3d})
