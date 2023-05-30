import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
#from copy import copy
import copy
import scipy
try:
    from utils.pc_utils_Norm import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                                    rotate_shape, random_rotate_one_axis,
                                    jitter_pointcloud_adaptive)
except:
    from utils_GAST.pc_utils_Norm import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                                    rotate_shape, random_rotate_one_axis,
                                    jitter_pointcloud_adaptive)
from tqdm import tqdm
try:
    import open3d as o3d # graspnet
    from data import transformations as trans
except:
    pass


try:
    from generate_c import deformation, noise, part
    from generate_c import background_noise,cutout
except:
    pass

eps = 10e-4
NUM_POINTS = 1024
#NUM_POINTS = 1500
idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}

def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)

def remove(points, norm_curv, p_keep=0.5):
    #p_keep = p_keep + np.random.rand(1) * (1-p_keep) # random sample
    if isinstance(p_keep, tuple):
        p_keep = p_keep[0] + np.random.rand(1) * (p_keep[1] - p_keep[0])
    rand_xyz = uniform_2_sphere()
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid

    dist_from_plane = np.dot(points_centered, rand_xyz)
    if p_keep == 0.5:
        mask = dist_from_plane > 0
    else:
        mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

    return points[mask, :], norm_curv[mask]

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


def load_data(data_path='data/modelnet40_c',corruption='cutout',severity=1,
        num_classes=40):
    if corruption == 'original':
        DATA_DIR = os.path.join(data_path, 'data_' + corruption + '.npy')
    else:
        DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)

    if num_classes == 40:
        return all_data, all_label

    pointda_label_dict = {
        1: 0, # bathtub
        2: 1, # bed
        4: 2, # bookshelf
        23: 3, # night_stand(cabinet)
        8: 4, # chair
        19: 5, # lamp
        22: 6, # monitor
        26: 7, # plant
        30: 8, # sofa
        33: 9, # table
    }

    pointda_label = [1, 2, 4, 8, 19, 22, 23, 26, 30, 33] # 1: bathtub, 2: bed, 4: bookshelf, 8: chair, 19: lamp, 22: monitor, 23: night_stand(cabinet), 26: plant, 30: sofa, 33: table
    pointda_indices = np.isin(all_label, pointda_label).squeeze(-1)
    all_data = all_data[pointda_indices, :, :]
    all_label = all_label[pointda_indices, :]
    all_label = np.array([pointda_label_dict[idx] for idx in all_label.squeeze()])

    return all_data, all_label


def load_modelnet_h5(partition='train'):
    BASE_DIR = './' #os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name.strip(), 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.5, high=0.5, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

class ModelNet40C(Dataset):
    def __init__(self, split='test', test_data_path='data/modelnet40_c/',
            corruption='background', severity=5, num_classes=40,
            random_scale=False, random_rotation=True, random_trans=False,
            rotate=True, subsample=1024, aug=False, corrupt_ori=False,
            mixed_corruption=False):
        if corruption != 'original':
            assert split == 'test'
        self.split = split
        self.corrupt_ori = corrupt_ori
        self.mixed_corruption = mixed_corruption
        self.aug = aug
        self.data_path = {
            "train": 'data/modelnet40_ply_hdf5_2048/',
            "val": 'data/modelnet40_ply_hdf5_2048/',
            "test":  test_data_path
        }[self.split]
        self.corruption = corruption
        if self.corrupt_ori:
            corruption = 'original'
        self.severity = int(severity)
        self.random_scale = random_scale
        self.random_rotation = random_rotation
        if corruption == 'original' and split != 'test':
            self.data, self.label = load_modelnet_h5(partition=split)
        else:
            self.data, self.label = load_data(self.data_path, corruption,
                self.severity, num_classes=num_classes)
        # self.num_points = num_points
        self.partition =  split
        self.random_trans = random_trans
        self.rotate = rotate
        self.random_fill = False
        self.subsample = subsample

    def __getitem__(self, item):
        pointcloud = self.data[item][:, :3]
        norm_curv = self.data[item][:, 3:]

        if self.mixed_corruption:
            pointcloud = background_noise(cutout(pointcloud, self.severity),
                    self.severity)
        elif self.corrupt_ori:
            t1, t2 = random.sample([deformation, noise, part], 2)
            f1 = random.choice(t1)
            f2 = random.choice(t2)
            #pointcloud = f1(f2(pointcloud, self.severity), self.severity)
            pointcloud = f1(pointcloud, int(self.severity))
            #f2(pointcloud, self.severity), self.severity)

        #print("mean", pointcloud.mean(axis=0))
        #print("norm", (pointcloud * pointcloud).sum(-1).max())
        #input()

        if len(pointcloud) > self.subsample: # and 'occlusion' in self.corruption:
            #print(len(pointcloud))
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud,
                    norm_curv, self.subsample)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        N = len(pointcloud)
        mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        mask[N:] = 0

        if self.corrupt_ori or ('cutout' in self.corruption or 'occlusion' in
                self.corruption or 'lidar' in self.corruption):
            dup_points = np.sum(np.power((pointcloud[None, :, :] - pointcloud[:,
                None, :]), 2),
                    axis=-1) < 1e-8
            dup_points[np.arange(len(pointcloud)), np.arange(len(pointcloud))] = False
            if np.any(dup_points):
                row, col = dup_points.nonzero()
                row, col = row[row<col], col[row<col]
                dup = np.unique(col)
                mask[dup] = 0
                pointcloud = pointcloud[mask.flatten()[:len(pointcloud)] > 0]
            valid, = mask.flatten().nonzero()
            #print("filter dup", len(valid))

        label = self.label[item]
        pc_ori = pointcloud.copy()
        ind = np.arange(len(pointcloud))
        N = len(pointcloud)
        if self.rotate:
            pointcloud = scale(pointcloud, 'unit_std')
            pointcloud = self.rotate_pc(pointcloud)
            if self.random_rotation:
                pointcloud = random_rotate_one_axis(pointcloud, "z")
        if self.aug:
            pointcloud = translate_pointcloud(pointcloud)
        if self.random_scale:
            random_scale = np.random.uniform(0.9, 1.1)
            pointcloud = random_scale * pointcloud
        #mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        if NUM_POINTS > N and self.random_fill:
            pointcloud = np.concatenate(
                    (pointcloud, np.random.randn(NUM_POINTS - N,
                        3).astype(pointcloud.dtype)), axis=0)
        else:
            #while 'lidar' not in self.corruption and len(pointcloud) < NUM_POINTS:
            if self.subsample < 2048:
                while len(pointcloud) < NUM_POINTS:
                    chosen = np.arange(N) #int(len(pointcloud)))
                    np.random.shuffle(chosen)
                    chosen = chosen[:NUM_POINTS - len(pointcloud)]
                    pointcloud = np.concatenate((
                        pointcloud,
                        pointcloud[chosen] #:NUM_POINTS - len(pointcloud)]
                    ), axis=0)
                    ind = np.concatenate((ind, chosen), axis=0)
                    assert len(pointcloud) == len(ind)
                    norm_curv = np.concatenate((
                        norm_curv,
                        norm_curv[chosen] #:NUM_POINTS - len(norm_curv)]
                    ), axis=0)
        #if pointcloud.shape[0] > NUM_POINTS:
        #    pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
        #    pc_ori = np.swapaxes(np.expand_dims(pc_ori, 0), 1, 2)
        #    #norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
        #    _, pointcloud, pc_ori = farthest_point_sample_np(pointcloud,
        #            pc_ori, NUM_POINTS)
        #    pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
        #    pc_ori = np.swapaxes(pc_ori.squeeze(), 1, 0).astype('float32')
        #    #norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')


        pointcloud = pointcloud
        if self.random_trans:
            random_trans = np.clip(self.random_trans*np.random.randn(1,3), -1*self.random_trans,
                    self.random_trans)
            pointcloud += random_trans
            return (pointcloud, label, random_trans)
        return (pointcloud, label, self.rotate_pc(pointcloud, True), mask, ind)

    def __len__(self):
        return self.data.shape[0]

    def rotate_pc(self, pointcloud, reverse=False):
        if reverse:
            pointcloud = rotate_shape(pointcloud, 'x', np.pi / 2)
            return pointcloud
        pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class ElasticDistortion:
    """Apply elastic distortion on sparse coordinate space. First projects the position onto a
    voxel grid and then apply the distortion to the voxel grid.
    Parameters
    ----------
    granularity: List[float]
        Granularity of the noise in meters
    magnitude:List[float]
        Noise multiplier in meters
    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(
        self, apply_distorsion= True, granularity = [0.2, 0.8], magnitude=[0.4, 1.6],
    ):
        assert len(magnitude) == len(granularity)
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        if not isinstance(coords, np.ndarray):
            coords = coords.numpy()
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim.tolist(), 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords = coords + interp(coords) * magnitude
        return coords #torch.tensor(coords).float()

    def __call__(self, data):
        # coords = data.pos / self._spatial_resolution
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    data = ElasticDistortion.elastic_distortion(data, self._granularity[i], self._magnitude[i],)
        return data

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={}, magnitude={})".format(
            self.__class__.__name__, self._apply_distorsion, self._granularity, self._magnitude,
        )



class ScanNet(Dataset):
    """
    scannet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot='data', partition='train', random_rotation=True,
            jitter=True, scale=1, scale_mode='unit_std', random_scale=False,
            zero_mean=True, elastic_distortion=False, self_distillation=False,
            subsample=1024,
            **kwargs):
        self.self_distillation = self_distillation
        if elastic_distortion:
            self.dist = ElasticDistortion(apply_distorsion=True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6])
        self.partition = partition
        if partition == 'val' or partition == 'train_all':
            partition = 'train'
        self.random_rotation = random_rotation
        self.jitter = jitter
        self.scale_mode = scale_mode
        self.scale = scale
        self.random_scale = random_scale
        # read data
        self.data, self.label = load_data_h5py_scannet10(partition, dataroot)
        self.num_examples = self.data.shape[0]
        io.cprint("number of " + partition + " examples in scannet" + ": " + str(self.data.shape[0]))
        self.data = list(self.data)
        self.subsample = subsample

        self.zero_mean = zero_mean

        # split train to train part and validation part
        if partition == "train" and self.partition != 'train_all':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if
                i % 10 < 8]).astype(int)
            #np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i
                % 10 >= 8]).astype(int)
            #np.random.shuffle(self.val_ind)

        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))
        # remove dup point
        processed_fn = f"data/PointDA_data/scannet/processed_{partition}.pt"
        if not os.path.exists(processed_fn):
            print("Remove dup point")
            for item in tqdm(range(len(self.data))):
                pointcloud = np.copy(self.data[item])[:, :3]
                norm_curv = np.copy(self.data[item])[:, 3:].astype(np.float32)
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
                self.data[item] = self.data[item][mask]
            print("Remove dup point done")
            self.data = np.array(self.data, dtype=object)

            with open(processed_fn, 'wb') as f:
                np.save(f, self.data, allow_pickle=True)
        else:
            with open(processed_fn, 'rb') as f:
                self.data = np.load(f, allow_pickle=True)
            print("Processed PCs are loaded")


    def __getitem__(self, item):
        item_ = item
        if self.partition == 'train':
            item = self.train_ind[item]
        elif self.partition == 'val':
            item = self.val_ind[item]
        pointcloud = np.copy(self.data[item])[:, :3]
        norm_curv = np.copy(self.data[item])[:, 3:].astype(np.float32)
        label = np.copy(self.label[item])

        #dup_points = np.sum(np.power((pointcloud[None, :, :] - pointcloud[:,
        #    None, :]), 2),
        #        axis=-1) < 1e-8
        #dup_points[np.arange(len(pointcloud)), np.arange(len(pointcloud))] = False
        #mask = np.ones(max(NUM_POINTS, len(pointcloud)))
        #ind = np.arange(len(pointcloud))
        #if np.any(dup_points):
        #    print("dup")
        #    row, col = dup_points.nonzero()
        #    row, col = row[row<col], col[row<col]
        #    dup = np.unique(col)
        #    mask[dup] = 0
        #    ind[col] = row
        #valid, = mask.nonzero()
        #mask = mask.astype(bool)
        #pointcloud = pointcloud[mask]
        #norm_curv = norm_curv[mask]

        # Rotate ScanNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > self.subsample:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud,
                    norm_curv, self.subsample)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        N = len(pointcloud)
        mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        mask[N:] = 0
        ind = np.arange(len(pointcloud))

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
            ind = np.concatenate((ind, chosen), axis=0)
            norm_curv = np.concatenate((
                norm_curv,
                norm_curv[chosen] #:NUM_POINTS - len(norm_curv)]
            ), axis=0)

        if self.self_distillation:
            pointcloud_aug = copy.deepcopy(pointcloud)
            if hasattr(self, 'dist'):
                pointcloud_aug = self.dist(pointcloud_aug)
        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:  #TODO 여기 살리기. 지금은 shapenet 학습한 모델이랑 맞추기 위함..
            pointcloud = random_rotate_one_axis(pointcloud, "z")
            if self.self_distillation:
                pointcloud_aug = random_rotate_one_axis(pointcloud_aug, "z")
        if self.jitter: # and (self.partition == 'train' and item not in
            #self.val_ind):
            pointcloud = jitter_pointcloud(pointcloud)
            if self.self_distillation:
                pointcloud_aug = jitter_pointcloud(pointcloud_aug)
            #pointcloud = jitter_pointcloud_adaptive(pointcloud)

        #pointcloud = pointcloud - pointcloud.mean(0, keepdims=True)
        if self.zero_mean:
            if self.scale_mode != 'unit_norm':
                pointcloud = self.scale * scale(pointcloud, self.scale_mode)
            else:
                pointcloud = self.scale * scale_to_unit_cube(pointcloud)
        if self.random_scale:
            random_scale = np.random.uniform(0.8, 1.2)
            if not self.self_distillation:
                pointcloud = random_scale * pointcloud
            else:
                pointcloud_aug = random_scale * pointcloud_aug

        #pointcloud = scale(pointcloud)
        if self.self_distillation:
            return (pointcloud, label, norm_curv, pointcloud_aug,
                    np.array(item_))

        return (pointcloud, label, norm_curv, min(ori_len, NUM_POINTS),
                ori_pointcloud, mask, ind)

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_ind)
        elif self.partition == 'val':
            return len(self.val_ind)
        return len(self.data)

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
            random_scale=False, zero_mean=True, random_remove=False,
            elastic_distortion=False, self_distillation=False, p_keep=0.7,
            subsample=1024):
        self.self_distillation = self_distillation
        self.p_keep = p_keep
        if elastic_distortion:
            self.dist = ElasticDistortion(apply_distorsion=True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6])
        self.random_remove = random_remove
        self.partition = partition
        if partition == 'val' or partition == 'train_all':
            partition = 'train'
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.jitter = jitter
        self.scale = scale
        self.scale_mode = scale_mode
        self.pc_list = []
        self.lbl_list = []
        self.subsample = subsample
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
        #if partition == "train":
        if partition == "train" and self.partition != 'train_all':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if
                i % 10 < 8]).astype(int)
            #np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i
                % 10 >= 8]).astype(int)
            #np.random.shuffle(self.val_ind)
        try:
            io.cprint("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            io.cprint("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))
        except:
            print("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            print("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        item_ = item
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
        if self.random_remove:
            pointcloud_rm, norm_curv_rm = remove(pointcloud, norm_curv,
                    self.p_keep)
            mean_rm = pointcloud_rm.mean(axis=0)
            std_rm = pointcloud_rm.reshape(-1).std(axis=0)
        else:
            pointcloud_rm = None
        if self.self_distillation:
            if self.scale_mode != 'unit_norm':
                if self.random_remove:
                    pointcloud_rm = self.scale * scale(pointcloud_rm, self.scale_mode)
            else:
                if self.random_remove:
                    pointcloud_rm = self.scale * scale_to_unit_cube(pointcloud_rm)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > self.subsample: #NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud,
                    norm_curv, self.subsample)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')

        N = len(pointcloud)
        mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        mask[N:] = 0
        ind = np.arange(len(pointcloud))

        if self.random_remove:
            if pointcloud_rm.shape[0] > NUM_POINTS:
                pointcloud_rm = np.swapaxes(np.expand_dims(pointcloud_rm, 0), 1, 2)
                norm_curv_rm = np.swapaxes(np.expand_dims(norm_curv_rm, 0), 1, 2)
                _, pointcloud_rm, norm_curv_rm = \
                    farthest_point_sample_np(pointcloud_rm, norm_curv_rm, NUM_POINTS)
                pointcloud_rm = np.swapaxes(pointcloud_rm.squeeze(), 1, 0).astype('float32')
                norm_curv_rm = np.swapaxes(norm_curv_rm.squeeze(), 1, 0).astype('float32')

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
            ind = np.concatenate((ind, chosen), axis=0)
            assert len(pointcloud) == len(ind)
        if self.random_remove:
            N = len(pointcloud_rm)
            while len(pointcloud_rm) < NUM_POINTS:
                chosen = np.arange(N) #int(len(pointcloud)))
                np.random.shuffle(chosen)
                chosen = chosen[:NUM_POINTS - len(pointcloud_rm)]
                pointcloud_rm = np.concatenate((
                    pointcloud_rm,
                    pointcloud_rm[chosen] #:NUM_POINTS - len(pointcloud)]
                ), axis=0)
                norm_curv_rm = np.concatenate((
                    norm_curv_rm,
                    norm_curv_rm[chosen] #:NUM_POINTS - len(norm_curv)]
                ), axis=0)
        if self.self_distillation:
            pointcloud_aug = copy.deepcopy(pointcloud_rm) if self.random_remove else copy.deepcopy(pointcloud)
            if hasattr(self, 'dist'):
                pointcloud_aug = self.dist(pointcloud_aug)

        # apply data rotation and augmentation on train samples
        if self.random_rotation==True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
            if self.random_remove:
                pointcloud_rm = random_rotate_one_axis(pointcloud_rm, "z")
            if self.self_distillation:
                pointcloud_aug = random_rotate_one_axis(pointcloud_aug, "z")
        if self.jitter: # and self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)
            if self.self_distillation:
                pointcloud_aug = jitter_pointcloud(pointcloud_aug)
            if self.random_remove:
                pointcloud_rm = jitter_pointcloud(pointcloud_rm)
            #pointcloud = jitter_pointcloud_adaptive(pointcloud)
        #pointcloud = pointcloud - pointcloud.mean(0, keepdims=True)
        if self.zero_mean:
            if self.scale_mode != 'unit_norm':
                pointcloud = self.scale * scale(pointcloud, self.scale_mode)
                if self.random_remove:
                    pointcloud_rm = self.scale * scale(pointcloud_rm, self.scale_mode)
            else:
                pointcloud = self.scale * scale_to_unit_cube(pointcloud)
                if self.random_remove:
                    pointcloud_rm = self.scale * scale_to_unit_cube(pointcloud_rm)
        if self.random_scale:
            random_scale = np.random.uniform(0.8, 1.2)
            if not self.self_distillation:
                pointcloud = random_scale * pointcloud
            else:
                pointcloud_aug = random_scale * pointcloud_aug
            if self.random_remove:
                random_scale = np.random.uniform(0.8, 1.2)
                pointcloud_rm = random_scale * pointcloud_rm
        if self.self_distillation:
            return (pointcloud, label, norm_curv, pointcloud_aug,
                    np.array(item_))
        if self.random_remove:
            return (pointcloud, label, norm_curv, NUM_POINTS, pointcloud_rm,
                    mean_rm, std_rm)
        return (pointcloud, label, norm_curv, NUM_POINTS, mask, ind)

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
            random_scale=False, zero_mean=True, random_remove=False,
            elastic_distortion=False, self_distillation=False, p_keep=0.7,
            subsample=1024):
        self.p_keep = p_keep
        self.self_distillation = self_distillation
        if elastic_distortion:
            self.dist = ElasticDistortion(apply_distorsion=True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6])
        self.random_remove = random_remove
        self.partition = partition
        if partition == 'val' or partition == 'train_all':
            partition = 'train'
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.pc_list = []
        self.lbl_list = []
        self.scale = scale
        self.scale_mode = scale_mode
        self.zero_mean = zero_mean
        self.subsample = subsample
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
        #if partition == "train":
        if partition == "train" and self.partition != 'train_all':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if
                i % 10 < 8]).astype(int)
            #np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i
                % 10 >= 8]).astype(int)
            #np.random.shuffle(self.val_ind)
        if io is not None:
            io.cprint("number of " + partition + " examples in shapenet: " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            io.cprint("Occurrences count of classes in shapenet " + partition + " set: " + str(dict(zip(unique, counts))))
        else:
            print("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
            unique, counts = np.unique(self.label, return_counts=True)
            print("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        item_ = item
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

        if self.random_remove:
            pointcloud_rm, norm_curv_rm = remove(pointcloud, norm_curv,
                    self.p_keep)
            mean_rm = pointcloud_rm.mean(axis=0)
            std_rm = pointcloud_rm.reshape(-1).std(axis=0)
        else:
            pointcloud_rm = None
        if self.self_distillation:
            if self.scale_mode != 'unit_norm':
                if self.random_remove:
                    pointcloud_rm = self.scale * scale(pointcloud_rm, self.scale_mode)
            else:
                if self.random_remove:
                    pointcloud_rm = self.scale * scale_to_unit_cube(pointcloud_rm)
        # Rotate ShapeNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud, label)
        if self.random_remove:
            pointcloud_rm = self.rotate_pc(pointcloud_rm, label)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > self.subsample:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(pointcloud,
                    norm_curv, self.subsample)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype('float32')
        if self.random_remove:
            if pointcloud_rm.shape[0] > NUM_POINTS:
                pointcloud_rm = np.swapaxes(np.expand_dims(pointcloud_rm, 0), 1, 2)
                norm_curv_rm = np.swapaxes(np.expand_dims(norm_curv_rm, 0), 1, 2)
                _, pointcloud_rm, norm_curv_rm = \
                    farthest_point_sample_np(pointcloud_rm, norm_curv_rm, NUM_POINTS)
                pointcloud_rm = np.swapaxes(pointcloud_rm.squeeze(), 1, 0).astype('float32')
                norm_curv_rm = np.swapaxes(norm_curv_rm.squeeze(), 1, 0).astype('float32')

        N = len(pointcloud)
        mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        mask[N:] = 0
        ind = np.arange(len(pointcloud))
        while len(pointcloud) < NUM_POINTS:
            chosen = np.arange(N) #int(len(pointcloud)))
            np.random.shuffle(chosen)
            chosen = chosen[:NUM_POINTS - len(pointcloud)]
            pointcloud = np.concatenate((
                pointcloud,
                pointcloud[chosen] #:NUM_POINTS - len(pointcloud)]
            ), axis=0)
            ind = np.concatenate((ind, chosen), axis=0)
            assert len(pointcloud) == len(ind)
            norm_curv = np.concatenate((
                norm_curv,
                norm_curv[chosen] #:NUM_POINTS - len(norm_curv)]
            ), axis=0)
        assert len(pointcloud) == NUM_POINTS
        assert len(norm_curv) == NUM_POINTS
        if self.random_remove:
            N = len(pointcloud_rm)
            while len(pointcloud_rm) < NUM_POINTS:
                chosen = np.arange(N) #int(len(pointcloud)))
                np.random.shuffle(chosen)
                chosen = chosen[:NUM_POINTS - len(pointcloud_rm)]
                pointcloud_rm = np.concatenate((
                    pointcloud_rm,
                    pointcloud_rm[chosen] #:NUM_POINTS - len(pointcloud)]
                ), axis=0)
                norm_curv_rm = np.concatenate((
                    norm_curv_rm,
                    norm_curv_rm[chosen] #:NUM_POINTS - len(norm_curv)]
                ), axis=0)
        if self.self_distillation:
            pointcloud_aug = copy.deepcopy(pointcloud_rm) if self.random_remove else copy.deepcopy(pointcloud)
            if hasattr(self, 'dist'):
                pointcloud_aug = self.dist(pointcloud_aug)
        # apply data rotation and augmentation on train samples
        if self.random_rotation == True:
            pointcloud = random_rotate_one_axis(pointcloud, "z")
            if self.random_remove:
                pointcloud_rm = random_rotate_one_axis(pointcloud_rm, "z")
            if self.self_distillation:
                pointcloud_aug = random_rotate_one_axis(pointcloud_aug, "z")
        if self.jitter and self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(pointcloud)
            if self.self_distillation:
                pointcloud_aug = jitter_pointcloud(pointcloud_aug)
            if self.random_remove:
                pointcloud_rm = jitter_pointcloud(pointcloud_rm)
            #pointcloud = jitter_pointcloud_adaptive(pointcloud)
        #pointcloud = pointcloud - pointcloud.mean(0, keepdims=True)
        if self.zero_mean:
            if self.scale_mode != 'unit_norm':
                pointcloud = self.scale * scale(pointcloud, self.scale_mode)
                if self.random_remove:
                    pointcloud_rm = self.scale * scale(pointcloud_rm, self.scale_mode)
            else:
                pointcloud = self.scale * scale_to_unit_cube(pointcloud)
                if self.random_remove:
                    pointcloud_rm = self.scale * scale_to_unit_cube(pointcloud_rm)
        if self.random_scale:
            random_scale = np.random.uniform(0.8, 1.2)
            if not self.self_distillation:
                pointcloud = random_scale * pointcloud
            else:
                pointcloud_aug = random_scale * pointcloud_aug
            if self.random_remove:
                random_scale = np.random.uniform(0.8, 1.2)
                pointcloud_rm = random_scale * pointcloud_rm
        if self.self_distillation:
            return (pointcloud, label, norm_curv, pointcloud_aug,
                    np.array(item_))
        if self.random_remove:
            return (pointcloud, label, norm_curv, NUM_POINTS, pointcloud_rm,
                    mean_rm, std_rm)
        return (pointcloud, label, norm_curv, NUM_POINTS, mask, ind)

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




####### GraspNet

class GraspNetPointClouds(Dataset):
    def __init__(self, dataroot, partition='train'):
        super(GraspNetPointClouds).__init__()
    def __getitem__(self, item):
        o3d_pointcloud = o3d.io.read_point_cloud(self.pc_list[item])
        pointcloud = np.asarray(o3d_pointcloud.points)

        pointcloud = pointcloud.astype(np.float32)
        path = self.pc_list[item].split('.x')[0]
        label = np.copy(self.label[item])

        data_item = {}
        data_item['PC'] = pointcloud
        data_item['Label'] = label
        data_item['Paths'] = path

        return data_item

    def __len__(self):
        return len(self.pc_list)

    def get_data_loader(self, batch_size, num_workers, drop_last, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)

class GraspNetRealPointClouds(GraspNetPointClouds):
    def __init__(self, dataroot, mode, partition='train'):
        super(GraspNetRealPointClouds).__init__()
        self.partition = partition

        DATA_DIR = os.path.join(dataroot, partition, "Real", mode) # mode can be 'kinect' or 'realsense'
        # read data
        xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))

        self.pc_list = []
        self.lbl_list = []

        for xyz_path in xyzs_list:
            self.pc_list.append(xyz_path)
            self.lbl_list.append(int(xyz_path.split('/')[-2]))

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

class GraspNetSynthetictPointClouds(GraspNetPointClouds):
    def __init__(self, dataroot, partition='train', device=None):
        super(GraspNetSynthetictPointClouds).__init__()
        self.partition = partition

        if device == None:
            DATA_DIR_kinect = os.path.join(dataroot, partition, "Synthetic", "kinect")
            DATA_DIR_realsense = os.path.join(dataroot, partition, "Synthetic", "realsense")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR_kinect, '*', '*.xyz')))
            xyzs_list_realsense = sorted(glob.glob(os.path.join(DATA_DIR_realsense, '*', '*.xyz')))

            xyzs_list.extend(xyzs_list_realsense)
        elif device == 'kinect':
            DATA_DIR = os.path.join(dataroot, partition, "Synthetic", "kinect")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))
        elif device == 'realsense':
            DATA_DIR = os.path.join(dataroot, partition, "Synthetic", "realsense")
            xyzs_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.xyz')))

        self.pc_list = []
        self.lbl_list = []

        for xyz_path in xyzs_list:
            self.pc_list.append(xyz_path)
            self.lbl_list.append(int(xyz_path.split('/')[-2]))

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)
