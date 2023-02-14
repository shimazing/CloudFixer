import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from copy import copy
try:
    from utils.pc_utils_Norm import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                                    rotate_shape, random_rotate_one_axis,
                                    jitter_pointcloud_adaptive)
except:
    from utils_GAST.pc_utils_Norm import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                                    rotate_shape, random_rotate_one_axis,
                                    jitter_pointcloud_adaptive)
from tqdm import tqdm

eps = 10e-4
NUM_POINTS = 1024
idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}

def scale(pc, scale_mode):
    pc = pc - pc.mean(0, keepdims=True)
    if scale_mode == 'unit_std':
        pc /= np.std(pc)
    elif scale_mode == 'unit_val':
        #pc /= pc.abs().max()
        pc /= np.amax(np.abs(pc)) # .absolute().max()

    return pc

def load_data_h5py_scannet10(partition, dataroot='data'):
    """
    Input:
        partition - train/test
    Return:
        data,label arrays
    """
    #DATA_DIR = dataroot +
    DATA_DIR = os.path.join(dataroot, 'PointDA_data/scannet_norm_curv_angle')
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, '%s_*.h5' % partition))):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:]
        label = f['label'][:]
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return np.array(all_data).astype('float32'), np.array(all_label).astype('int64')


class ScanNet(Dataset):
    """
    scannet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot='data', partition='train', random_rotation=True,
            jitter=True, scale=1, scale_mode='unit_std', random_scale=False,
            zero_mean=True):
        self.partition = partition
        if partition == 'val':
            partition = 'train'
        self.random_rotation = random_rotation
        self.jitter = jitter
        self.scale_mode = scale_mode
        self.scale = scale
        self.random_scale = random_scale
        # read data
        self.data, self.label = load_data_h5py_scannet10(partition, dataroot)
        self.num_examples = self.data.shape[0]
        self.zero_mean = zero_mean

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if
                i % 10 < 8]).astype(int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i
                % 10 >= 8]).astype(int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scannet" + ": " + str(self.data.shape[0]))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        if self.partition == 'train':
            item = self.train_ind[item]
        elif self.partition == 'val':
            item = self.val_ind[item]
        pointcloud = np.copy(self.data[item])[:, :3]
        norm_curv = np.copy(self.data[item])[:, 3:].astype(np.float32)
        label = np.copy(self.label[item])

        dup_points = np.sum(np.power((pointcloud[None, :, :] - pointcloud[:,
            None, :]), 2),
                axis=-1) < 1e-8
        dup_points[np.arange(len(pointcloud)), np.arange(len(pointcloud))] = False
        mask = np.ones(len(pointcloud))
        if np.any(dup_points):
            row, col = dup_points.nonzero()
            row, col = row[row<col], col[row<col]
            dup = np.unique(col)
            mask[dup] = 0
        valid, = mask.nonzero()
        mask = mask.astype(bool)
        pointcloud = pointcloud[mask]
        norm_curv = norm_curv[mask]

        # Rotate ScanNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud, norm_curv, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        if self.scale_mode != 'unit_norm':
            pointcloud = self.scale * scale(pointcloud, self.scale_mode)
        else:
            pointcloud = self.scale * scale_to_unit_cube(pointcloud) #, mask)

        ori_len = len(pointcloud)
        ori_pointcloud = np.concatenate((pointcloud,
            np.zeros((NUM_POINTS - len(pointcloud),3))), axis=0)

        #if len(pointcloud) < NUM_POINTS: # uniformly duplicate
        N = len(pointcloud)
        while len(pointcloud) < NUM_POINTS:
            chosen = np.arange(N) #int(len(pointcloud)))
            np.random.shuffle(chosen)
            chosen = chosen[:NUM_POINTS - len(pointcloud)]
            pointcloud = np.concatenate((
                pointcloud,
                pointcloud[chosen] #:NUM_POINTS - len(pointcloud)]
            ), axis=0)
            norm_curv = np.concatenate((
                norm_curv,
                norm_curv[chosen] #:NUM_POINTS - len(norm_curv)]
            ), axis=0)


        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:  #TODO 여기 살리기. 지금은 shapenet 학습한 모델이랑 맞추기 위함..
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.jitter: # and (self.partition == 'train' and item not in
            #self.val_ind):
            pointcloud = jitter_pointcloud(pointcloud)
            #pointcloud = jitter_pointcloud_adaptive(pointcloud)

        #pointcloud = pointcloud - pointcloud.mean(0, keepdims=True)
        if self.zero_mean:
            if self.scale_mode != 'unit_norm':
                pointcloud = self.scale * scale(pointcloud, self.scale_mode)
            else:
                pointcloud = self.scale * scale_to_unit_cube(pointcloud)
        if self.random_scale:
            random_scale = np.random.uniform(0.8, 1.2)
            pointcloud = random_scale * pointcloud

        #pointcloud = scale(pointcloud)

        return (pointcloud, label, norm_curv, min(ori_len, NUM_POINTS),
                ori_pointcloud)

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_ind)
        elif self.partition == 'val':
            return len(self.val_ind)
        return self.data.shape[0]

    # scannet is rotated such that the up direction is the y axis
    def rotate_pc(self, pointcloud):
        pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class ModelNet(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot='./data', partition='train',
            random_rotation=True, jitter=True, scale=1, scale_mode='unit_std',
            random_scale=False, zero_mean=True):
        self.partition = partition
        if partition == 'val':
            partition = 'train'
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.jitter = jitter
        self.scale = scale
        self.scale_mode = scale_mode
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "modelnet") #_norm_curv")
        self.zero_mean = zero_mean
        #DATA_DIR = 'data/PointDA_data/modelnet' #_norm_curv_angle'

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if
                i % 10 < 8]).astype(int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i
                % 10 >= 8]).astype(int)
            np.random.shuffle(self.val_ind)
        try:
            io.cprint("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            io.cprint("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))
        except:
            print("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            print("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        if self.partition == 'train':
            item = self.train_ind[item]
        elif self.partition == 'val':
            item = self.val_ind[item]
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        norm_curv = np.load(self.pc_list[item])[:, 3:].astype(np.float32)
        label = np.copy(self.label[item])
        if self.scale_mode != 'unit_norm':
            pointcloud = self.scale * scale(pointcloud, self.scale_mode)
        else:
            pointcloud = self.scale * scale_to_unit_cube(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud, norm_curv, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.jitter and self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)
            #pointcloud = jitter_pointcloud_adaptive(pointcloud)
        #pointcloud = pointcloud - pointcloud.mean(0, keepdims=True)
        if self.zero_mean:
            if self.scale_mode != 'unit_norm':
                pointcloud = self.scale * scale(pointcloud, self.scale_mode)
            else:
                pointcloud = self.scale * scale_to_unit_cube(pointcloud)
        if self.random_scale:
            random_scale = np.random.uniform(0.8, 1.2)
            pointcloud = random_scale * pointcloud
        return (pointcloud, label, norm_curv, NUM_POINTS)

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_ind)
        elif self.partition == 'val':
            return len(self.val_ind)
        return len(self.pc_list)


class ShapeNet(Dataset):
    """
    Sahpenet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot='data', partition='train', random_rotation=True,
            scale=1, val=False, jitter=True, scale_mode='unit_std',
            random_scale=False, zero_mean=True):
        self.partition = partition
        if partition == 'val':
            partition = 'train'
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.pc_list = []
        self.lbl_list = []
        self.scale = scale
        self.scale_mode = scale_mode
        self.zero_mean = zero_mean
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "shapenet") #_norm_curv")
        #DATA_DIR = 'data/PointDA_data/shapenet' #_norm_curv_angle'

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)
        self.val = val
        self.jitter = jitter
        #self.pointclouds = [np.load(self.pc_list[item])[:,
        #    :3].astype(np.float32) for item in tqdm(range(len(self.pc_list)))]
        #self.norm_curvs = [np.load(self.pc_list[item])[:,
        #    3:].astype(np.float32) for item in range(len(self.pc_list))]

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if
                i % 10 < 8]).astype(int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i
                % 10 >= 8]).astype(int)
            np.random.shuffle(self.val_ind)
        if io is not None:
            io.cprint("number of " + partition + " examples in shapenet: " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            io.cprint("Occurrences count of classes in shapenet " + partition + " set: " + str(dict(zip(unique, counts))))
        else:
            print("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            print("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        if self.partition == 'train':
            item = self.train_ind[item]
        elif self.partition == 'val':
            item = self.val_ind[item]
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        norm_curv = np.load(self.pc_list[item])[:, 3:].astype(np.float32)
        label = np.copy(self.label[item])
        if self.scale_mode != 'unit_norm':
            pointcloud = self.scale * scale(pointcloud, self.scale_mode)
        else:
            pointcloud = self.scale * scale_to_unit_cube(pointcloud)
        # Rotate ShapeNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud, label)

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud, norm_curv, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.jitter and self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)
            #pointcloud = jitter_pointcloud_adaptive(pointcloud)
        #pointcloud = pointcloud - pointcloud.mean(0, keepdims=True)
        if self.zero_mean:
            if self.scale_mode != 'unit_norm':
                pointcloud = self.scale * scale(pointcloud, self.scale_mode)
            else:
                pointcloud = self.scale * scale_to_unit_cube(pointcloud)
        if self.random_scale:
            random_scale = np.random.uniform(0.8, 1.2)
            pointcloud = random_scale * pointcloud

        return (pointcloud, label, norm_curv, NUM_POINTS)

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_ind)
        elif self.partition == 'val':
            return len(self.val_ind)
        #if self.val:
        #    return len(self.val_ind)
        return len(self.pc_list)


    # shpenet is rotated such that the up direction is the y axis in all shapes except plant
    def rotate_pc(self, pointcloud, label):
        if label.item(0) != label_to_idx["plant"]:
            pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1

    def __init__(self, path, cates, split, scale_mode='shape_bbox', transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        self.path = 'data/shapenet.hdf5'
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):

        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc), j, cate_name

        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):

                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return (data['pointcloud'], 0, 0) # dummy for compatibility
