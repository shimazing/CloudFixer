import os
import glob
import h5py
import random
from tqdm import tqdm

try:
    import open3d as o3d
except:
    print("unable to import open3d. GraspNet10 is not available")
import numpy as np
import scipy
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.pc_utils import *

NUM_POINTS = 1024


class ModelNet40C(Dataset):
    def __init__(self, args, partition):
        super().__init__()
        self.dataset = args.dataset
        self.partition = partition
        self.scenario = getattr(args, "scenario", "")
        self.subsample = getattr(args, "subsample", 2048)

        if len(args.dataset.split("_")) == 1:
            self.corruption = "original"
        elif len(args.dataset.split("_")) == 2:
            self.corruption = "_".join(args.dataset.split("_")[1:])
        else:
            self.corruption = "_".join(args.dataset.split("_")[1:-1])
        if self.corruption != "original":
            assert partition == "test"
            self.severity = args.dataset.split("_")[-1]

        self.rotate = args.rotate if hasattr(args, "rotate") else True

        # augmentation
        if partition in ["train", "train_all"]:
            self.jitter = args.jitter if hasattr(args, "jitter") else True
            self.random_scale = (
                args.random_scale if hasattr(args, "random_scale") else False
            )
            self.random_rotation = (
                args.random_rotation if hasattr(args, "random_rotation") else True
            )
            self.random_trans = (
                args.random_trans if hasattr(args, "random_trans") else False
            )
            self.aug = args.aug if hasattr(args, "aug") else False
        else:
            (
                self.jitter,
                self.random_scale,
                self.random_rotation,
                self.random_trans,
                self.aug,
            ) = (False, False, False, False, False)

        self.label_to_idx = {
            label: idx
            for idx, label in enumerate(
                [
                    "airplane",
                    "bathtub",
                    "bed",
                    "bench",
                    "bookshelf",
                    "bottle",
                    "bowl",
                    "car",
                    "chair",
                    "cone",
                    "cup",
                    "curtain",
                    "desk",
                    "door",
                    "dresser",
                    "flower_pot",
                    "glass_box",
                    "guitar",
                    "keyboard",
                    "lamp",
                    "laptop",
                    "mantel",
                    "monitor",
                    "night_stand",
                    "person",
                    "piano",
                    "plant",
                    "radio",
                    "range_hood",
                    "sink",
                    "sofa",
                    "stairs",
                    "stool",
                    "table",
                    "tent",
                    "toilet",
                    "tv_stand",
                    "vase",
                    "wardrobe",
                    "xbox",
                ]
            )
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        if self.corruption == "original":
            self.pc_list, self.label_list = self.load_modelnet40(
                args.dataset_dir, partition=partition
            )
        else:
            self.pc_list, self.label_list = self.load_modelnet40_c(
                args.dataset_dir, self.corruption, self.severity
            )

        # print dataset statistics
        unique, counts = np.unique(self.label_list, return_counts=True)
        print(
            f"number of {partition} examples in {args.dataset} : {str(len(self.pc_list))}"
        )
        print(
            f"Occurrences count of classes in {args.dataset} {partition} set: {str(dict(zip(unique, counts)))}"
        )

    def load_modelnet40(self, data_path, partition="train"):
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(data_path, f"ply_data_{partition}*.h5")):
            f = h5py.File(h5_name.strip(), "r")
            data = f["data"][:].astype("float32")
            label = f["label"][:].astype("int64")
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0).squeeze(-1)
        return all_data, all_label

    def load_modelnet40_c(
        self,
        data_path="data/modelnet40_c",
        corruption="cutout",
        severity=1,
        num_classes=40,
    ):
        if corruption == "original":
            data_dir = os.path.join(data_path, f"data_{corruption}.npy")
            all_data = np.load(data_dir)
            label_dir = os.path.join(data_path, "label.npy")
            all_label = np.load(label_dir).squeeze(-1)
        elif self.scenario == "mixed":
            corruption_list = [
                "background",
                "cutout",
                "density",
                "density_inc",
                "distortion",
                "distortion_rbf",
                "distortion_rbf_inv",
                "gaussian",
                "impulse",
                "lidar",
                "occlusion",
                "rotation",
                "shear",
                "uniform",
                "upsampling",
            ]
            data_dir_list = [
                os.path.join(data_path, f"data_{corruption}_{severity}.npy")
                for corruption in corruption_list
            ]
            all_data_list = [np.load(data_dir) for data_dir in data_dir_list]
            selected_indices = np.random.choice(
                len(corruption_list), len(all_data_list[0]), replace=True
            )
            all_data = []
            for iter_idx, corruption_idx in enumerate(selected_indices):
                pointcloud_norm_curv = all_data_list[corruption_idx][iter_idx]
                pointcloud = pointcloud_norm_curv[:, :3]
                norm_curv = pointcloud_norm_curv[:, 3:]

                # NUM_POINTS for all corruptions
                if pointcloud.shape[0] > NUM_POINTS:
                    pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
                    norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
                    _, pointcloud, norm_curv = farthest_point_sample_np(
                        pointcloud, norm_curv, NUM_POINTS
                    )
                    pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype(
                        "float32"
                    )
                    norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype("float32")
                N = len(pointcloud)
                mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
                mask[N:] = 0
                ind = np.arange(len(pointcloud))
                while len(pointcloud) < NUM_POINTS:
                    chosen = np.arange(N)
                    np.random.shuffle(chosen)
                    chosen = chosen[: NUM_POINTS - len(pointcloud)]
                    pointcloud = np.concatenate(
                        (pointcloud, pointcloud[chosen]), axis=0
                    )
                    ind = np.concatenate((ind, chosen), axis=0)
                    norm_curv = np.concatenate((norm_curv, norm_curv[chosen]), axis=0)

                all_data.append(pointcloud)
            all_data = np.array(all_data)
            label_dir = os.path.join(data_path, "label.npy")
            all_label = np.load(label_dir).squeeze(-1)
        else:
            data_dir = os.path.join(data_path, f"data_{corruption}_{severity}.npy")
            all_data = np.load(data_dir)
            label_dir = os.path.join(data_path, "label.npy")
            all_label = np.load(label_dir)  # .squeeze(-1)
            print(all_label.shape)
            if all_label.ndim == 2:
                all_label = all_label.squeeze(-1)

        if self.scenario == "temporally_correlated":
            sorted_indices = np.argsort(all_label)
            all_data = all_data[sorted_indices]
            all_label = all_label[sorted_indices]

        print(f"num_classes: {num_classes}")

        if num_classes == 40:
            return all_data, all_label

        pointda_label_dict = {
            1: 0,  # bathtub
            2: 1,  # bed
            4: 2,  # bookshelf
            23: 3,  # night_stand(cabinet)
            8: 4,  # chair
            19: 5,  # lamp
            22: 6,  # monitor
            26: 7,  # plant
            30: 8,  # sofa
            33: 9,  # table
        }
        pointda_label = [
            1,
            2,
            4,
            8,
            19,
            22,
            23,
            26,
            30,
            33,
        ]  # 1: bathtub, 2: bed, 4: bookshelf, 8: chair, 19: lamp, 22: monitor, 23: night_stand(cabinet), 26: plant, 30: sofa, 33: table
        pointda_indices = np.isin(all_label, pointda_label).squeeze(-1)
        all_data = all_data[pointda_indices, :, :]
        all_label = all_label[pointda_indices, :]
        all_label = np.array([pointda_label_dict[idx] for idx in all_label])
        return all_data, all_label

    def get_label_to_idx(self, args):
        npy_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, "*", "train", "*.npy"))
        )
        label_to_idx = {
            label: idx
            for idx, label in enumerate(
                list(np.unique([_dir.split("/")[-3] for _dir in npy_list]))
            )
        }
        return label_to_idx

    def __getitem__(self, item):
        pointcloud = self.pc_list[item][:, :3]
        label = self.label_list[item]
        mask = np.ones((len(pointcloud), 1)).astype(pointcloud.dtype)
        ind = np.arange(len(pointcloud))

        # identify duplicated points
        if (
            "occlusion" in self.corruption
            or "density_inc" in self.corruption
            or "lidar" in self.corruption
        ):
            dup_points = (
                np.sum(
                    np.power((pointcloud[None, :, :] - pointcloud[:, None, :]), 2),
                    axis=-1,
                )
                < 1e-8
            )
            dup_points[np.arange(len(pointcloud)), np.arange(len(pointcloud))] = False
            if np.any(dup_points):
                row, col = dup_points.nonzero()
                row, col = row[row < col], col[row < col]
                filter = (row.reshape(-1, 1) == col).astype(float).sum(-1) == 0
                row, col = row[filter], col[filter]
                ind[col] = row
                dup = np.unique(col)
                mask[dup] = 0

        if self.rotate:
            pointcloud = scale(pointcloud, "unit_std")
            pointcloud = rotate_pc(pointcloud)
            if self.random_rotation:
                pointcloud = random_rotate_one_axis(pointcloud, "z")

        if self.jitter:
            pointcloud = jitter_pointcloud(pointcloud)

        if mask.sum() > self.subsample:
            valid = mask.nonzero()[0]
            pointcloud_ = pointcloud[mask.flatten()[: len(pointcloud)] > 0]
            pointcloud_ = np.swapaxes(np.expand_dims(pointcloud_, 0), 1, 2)
            centroids, pointcloud_, _ = farthest_point_sample_np(
                pointcloud_, None, self.subsample
            )
            pointcloud_ = np.swapaxes(pointcloud_.squeeze(), 1, 0).astype("float32")
            centroids = centroids.squeeze()
            assert len(centroids) == self.subsample
            mask_ = np.zeros_like(mask)
            mask_[valid[centroids]] = 1  # reg줄  subsample 된 것! 나머지는
            assert np.all(mask[mask_ == 1] == 1)
            mask = mask_
            if self.corruption == "original":
                pointcloud = pointcloud[mask.squeeze(-1).astype(bool)]
                mask = mask[mask.squeeze(-1).astype(bool)]
                ind = np.arange(len(pointcloud))

        if self.subsample < 2048:
            valid = mask.nonzero()[0]
            while len(pointcloud) < NUM_POINTS:  # len(mask):# NUM_POINTS:
                np.random.shuffle(valid)
                chosen = valid[: NUM_POINTS - len(pointcloud)]
                pointcloud = np.concatenate(
                    (
                        pointcloud,
                        pointcloud[chosen],
                    ),
                    axis=0,
                )
                mask = np.concatenate((mask, np.zeros_like(mask[chosen])), axis=0)
                ind = np.concatenate((ind, chosen), axis=0)
                assert len(pointcloud) == len(ind)

        return (pointcloud, label, mask, ind)

    def __len__(self):
        return len(self.pc_list)


class PointDA10(Dataset):
    def __init__(self, args, partition):
        super().__init__()
        self.dataset = args.dataset
        self.partition = partition

        self.scale = args.scale if hasattr(args, "scale") else 1
        self.scale_mode = args.scale_mode if hasattr(args, "scale_mode") else "unit_std"
        self.zero_mean = args.zero_mean if hasattr(args, "zero_mean") else True

        # augmentation
        if partition in ["train", "train_all"]:
            self.random_rotation = (
                args.random_rotation if hasattr(args, "random_rotation") else True
            )
            self.random_remove = (
                args.random_remove if hasattr(args, "random_remove") else False
            )
            self.p_keep = args.p_keep if hasattr(args, "p_keep") else 0.7
            self.random_scale = (
                args.random_scale if hasattr(args, "random_scale") else False
            )
            self.jitter = args.jitter if hasattr(args, "jitter") else True
            self.elastic_distortion = (
                args.elastic_distortion
                if hasattr(args, "elastic_distortion")
                else False
            )
            if self.elastic_distortion:
                self.dist = ElasticDistortion(
                    apply_distorsion=True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6]
                )
            self.self_distillation = (
                args.self_distillation if hasattr(args, "self_distillation") else False
            )
        else:
            (
                self.random_rotation,
                self.random_remove,
                self.p_keep,
                self.random_scale,
                self.jitter,
                self.elastic_distortion,
                self.self_distillation,
            ) = (False, False, 1, False, False, False, False)

        self.label_to_idx = self.get_label_to_idx(args)
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.pc_list, self.label_list = self.get_data(args, partition)

        # print dataset statistics
        unique, counts = np.unique(self.label_list, return_counts=True)
        print(
            f"number of {partition} examples in {args.dataset} : {str(len(self.pc_list))}"
        )
        print(
            f"Occurrences count of classes in {args.dataset} {partition} set: {str(dict(zip(unique, counts)))}"
        )

    def get_label_to_idx(self, args):
        npy_list = sorted(
            glob.glob(os.path.join(args.dataset_dir, "*", "train", "*.npy"))
        )
        label_to_idx = {
            label: idx
            for idx, label in enumerate(
                list(np.unique([_dir.split("/")[-3] for _dir in npy_list]))
            )
        }
        return label_to_idx

    def get_data(self, args, partition):
        partition_dir = (
            "train" if partition in ["train", "train_all", "val"] else partition
        )
        if args.dataset in ["modelnet", "shapenet"]:
            pc_list, label_list = [], []
            npy_list = sorted(
                glob.glob(os.path.join(args.dataset_dir, "*", partition_dir, "*.npy"))
            )
            for _dir in npy_list:
                pc_list.append(_dir)
                label_list.append(self.label_to_idx[_dir.split("/")[-3]])
            pc_list = np.array(pc_list)
            label_list = np.asarray(label_list)
        else:
            pc_list, label_list = self.load_scannet(partition_dir, args.dataset_dir)
            # remove duplicated point
            processed_fn = f"{args.dataset_dir}/processed_{partition_dir}.pt"
            if not os.path.exists(processed_fn):
                print("Remove duplicated poin start")
                for item in tqdm(range(len(pc_list))):
                    pointcloud = np.copy(pc_list[item])[:, :3]
                    dup_points = (
                        np.sum(
                            np.power(
                                (pointcloud[None, :, :] - pointcloud[:, None, :]), 2
                            ),
                            axis=-1,
                        )
                        < 1e-8
                    )
                    dup_points[
                        np.arange(len(pointcloud)), np.arange(len(pointcloud))
                    ] = False
                    mask = np.ones(len(pointcloud))
                    if np.any(dup_points):
                        row, col = dup_points.nonzero()
                        row, col = row[row < col], col[row < col]
                        dup = np.unique(col)
                        mask[dup] = 0
                    mask = mask.astype(bool)
                    pc_list[item] = pc_list[item][mask]
                print("Remove duplicated point end")
                pc_list = np.array(pc_list, dtype=object)
                with open(processed_fn, "wb") as f:
                    np.save(f, pc_list, allow_pickle=True)
            else:
                with open(processed_fn, "rb") as f:
                    pc_list = np.load(f, allow_pickle=True)

        if partition == "train":
            idx_list = np.asarray(
                [i for i in range(len(pc_list)) if i % 10 < 8]
            ).astype(int)
        elif partition == "val":
            idx_list = np.asarray(
                [i for i in range(len(pc_list)) if i % 10 >= 8]
            ).astype(int)
        else:  # train_all, test
            idx_list = np.asarray(range(len(pc_list))).astype(int)
        return pc_list[idx_list], label_list[idx_list]

    def load_scannet(self, partition, dataset_dir="data"):
        """
        Input:
            partition - train/test
        Return:
            data,label arrays
        """
        all_data, all_label = [], []
        for h5_name in sorted(
            glob.glob(os.path.join(dataset_dir, f"{partition}_*.h5"))
        ):
            f = h5py.File(h5_name, "r")
            data = f["data"][:]
            label = f["label"][:]
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return np.array(all_data).astype("float32"), np.array(all_label).astype("int64")

    def __getitem__(self, item):
        pointcloud_norm_curv = (
            np.load(self.pc_list[item]).astype(np.float32)
            if isinstance(self.pc_list[item], str)
            else self.pc_list[item].astype(np.float32)
        )
        pointcloud = pointcloud_norm_curv[:, :3]
        norm_curv = pointcloud_norm_curv[:, 3:]
        label = np.copy(self.label_list[item])

        if (
            self.dataset == "shapenet" and label.item(0) != self.label_to_idx["plant"]
        ) or self.dataset == "scannet":
            pointcloud = rotate_pc(pointcloud)

        if self.scale_mode != "unit_norm":
            if self.random_remove:
                pointcloud_rm = self.scale * scale(pointcloud_rm, self.scale_mode)
        else:
            if self.random_remove:
                pointcloud_rm = self.scale * scale_to_unit_cube(pointcloud_rm)

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            norm_curv = np.swapaxes(np.expand_dims(norm_curv, 0), 1, 2)
            _, pointcloud, norm_curv = farthest_point_sample_np(
                pointcloud, norm_curv, NUM_POINTS
            )
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype("float32")
            norm_curv = np.swapaxes(norm_curv.squeeze(), 1, 0).astype("float32")

        N = len(pointcloud)
        mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        mask[N:] = 0
        ind = np.arange(len(pointcloud))

        while len(pointcloud) < NUM_POINTS:
            chosen = np.arange(N)
            np.random.shuffle(chosen)
            chosen = chosen[: NUM_POINTS - len(pointcloud)]
            pointcloud = np.concatenate((pointcloud, pointcloud[chosen]), axis=0)
            ind = np.concatenate((ind, chosen), axis=0)
            norm_curv = np.concatenate((norm_curv, norm_curv[chosen]), axis=0)

        # apply data rotation and augmentation on train samples
        if self.random_rotation:
            pointcloud = random_rotate_one_axis(pointcloud, "z")

        if self.jitter:
            pointcloud = jitter_pointcloud(pointcloud)

        return (pointcloud, label, mask, ind)

    def __len__(self):
        return len(self.pc_list)


class GraspNet10(Dataset):
    def __init__(self, args, partition):
        super().__init__()
        self.dataset = args.dataset
        self.partition = partition

        self.scale = args.scale if hasattr(args, "scale") else 1
        self.scale_mode = (
            args.scale_mode if hasattr(args, "scale_mode") else "unit_norm"
        )
        self.zero_mean = args.zero_mean if hasattr(args, "zero_mean") else False

        # augmentation
        if partition in ["train", "train_all"]:
            self.random_scale = (
                args.random_scale if hasattr(args, "random_scale") else True
            )
            self.random_rotation = (
                args.random_rotation if hasattr(args, "random_rotation") else False
            )
            self.self_distillation = (
                args.self_distillation if hasattr(args, "self_distillation") else False
            )
            self.jitter = args.jitter if hasattr(args, "jitter") else True
        else:
            (
                self.random_scale,
                self.random_rotation,
                self.self_distillation,
                self.jitter,
            ) = (False, False, False, False)

        self.label_to_idx = {
            label: idx
            for idx, label in enumerate(
                [
                    "bathtub",
                    "bed",
                    "bookshelf",
                    "cabinet",
                    "chair",
                    "lamp",
                    "monitor",
                    "plant",
                    "sofa",
                    "table",
                ]
            )
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.pc_list, self.label_list = self.get_data(args, partition)

        # print dataset statistics
        unique, counts = np.unique(self.label_list, return_counts=True)
        print(
            f"number of {partition} examples in {args.dataset} : {str(len(self.pc_list))}"
        )
        print(
            f"Occurrences count of classes in {args.dataset} {partition} set: {str(dict(zip(unique, counts)))}"
        )

    def get_data(self, args, partition):
        partition_dir = (
            "train" if partition in ["train", "train_all", "val"] else partition
        )
        if args.dataset == "synthetic":
            dataset_dir_kinect = os.path.join(
                args.dataset_dir, partition_dir, "Synthetic", "kinect"
            )
            dataset_dir_realsense = os.path.join(
                args.dataset_dir, partition_dir, "Synthetic", "realsense"
            )
            pc_list = sorted(glob.glob(os.path.join(dataset_dir_kinect, "*", "*.xyz")))
            pc_list.extend(
                sorted(glob.glob(os.path.join(dataset_dir_realsense, "*", "*.xyz")))
            )
        elif args.dataset == "kinect":
            dataset_dir_kinect = os.path.join(
                args.dataset_dir, partition_dir, "Real", "kinect"
            )
            pc_list = sorted(glob.glob(os.path.join(dataset_dir_kinect, "*", "*.xyz")))
        elif args.dataset == "realsense":
            dataset_dir_realsense = os.path.join(
                args.dataset_dir, partition_dir, "Real", "realsense"
            )
            pc_list = sorted(
                glob.glob(os.path.join(dataset_dir_realsense, "*", "*.xyz"))
            )
        pc_list = np.asarray(pc_list)
        label_list = np.asarray([int(pc.split("/")[-2]) for pc in pc_list])

        if partition == "train":
            idx_list = np.asarray(
                [i for i in range(len(pc_list)) if i % 10 < 8]
            ).astype(int)
        elif partition == "val":
            idx_list = np.asarray(
                [i for i in range(len(pc_list)) if i % 10 >= 8]
            ).astype(int)
        else:  # train_all, test
            idx_list = np.asarray(range(len(pc_list))).astype(int)
        return pc_list[idx_list], label_list[idx_list]

    def __getitem__(self, item):
        o3d_pointcloud = o3d.io.read_point_cloud(self.pc_list[item])
        pointcloud = np.asarray(o3d_pointcloud.points).astype(np.float32)
        label = self.label_list[item]

        if self.scale_mode != "unit_norm":
            pointcloud = self.scale * scale(pointcloud, self.scale_mode)
        else:
            pointcloud = self.scale * scale_to_unit_cube(pointcloud)

        N = len(pointcloud)
        mask = np.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        mask[N:] = 0
        ind = np.arange(len(pointcloud))

        if self.jitter:
            pointcloud = jitter_pointcloud(pointcloud)

        return (pointcloud, label, mask, ind)

    def __len__(self):
        return len(self.pc_list)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        dataset,
        labels=None,
        indices=None,
        num_samples=None,
        callback_get_label=None,
        imb_ratio=None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        if imb_ratio:
            selected_idx = np.random.choice(
                len(label_to_count), int(0.1 * len(label_to_count)), replace=False
            )
            for idx in selected_idx:
                weights[df["label"] == idx] *= imb_ratio
        self.weights = torch.DoubleTensor(weights.tolist())

    def _get_labels(self, dataset):
        return dataset.label_list

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class ElasticDistortion:
    def __init__(
        self, apply_distorsion=True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6]
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
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=0, fill_value=0
        )
        coords = coords + interp(coords) * magnitude
        return coords  # torch.tensor(coords).float()

    def __call__(self, data):
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    data = ElasticDistortion.elastic_distortion(
                        data,
                        self._granularity[i],
                        self._magnitude[i],
                    )
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(apply_distorsion={self._apply_distorsion}, granularity={self._granularity}, magnitude={self._magnitude})"
