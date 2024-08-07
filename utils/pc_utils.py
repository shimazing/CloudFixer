import os
import random

import numpy as np
import torch

eps = 10e-4
eps2 = 10e-6
KL_SCALER = 10.0
MIN_POINTS = 20
RADIUS = 0.5  # 0.5 * 3
NREGIONS = 3
NROTATIONS = 4
N = 16
K = 4
NUM_FEATURES = K * 3 + 1


def knn(x, k=5, mask=None, ind=None, return_dist=False):
    # mask : [B, N]
    # x : [B, C=3, N]
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = (
        -xx - inner - xx.transpose(2, 1)
    )  # 거리 가장 가까운거 골라야하니까 음수 붙여줌
    # B x N x N

    if ind is not None:
        # update mask only to consider duplicated points
        mask = ind == torch.arange(ind.shape[1]).to(ind.device)
    if mask is not None:
        B_ind, N_ind = (~mask).nonzero(as_tuple=True)
        pairwise_distance[B_ind, N_ind] = -np.inf
        pairwise_distance[B_ind, :, N_ind] = -np.inf
    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    if return_dist:
        B = x.shape[0]
        N = x.shape[2]
        dist = (
            -pairwise_distance[
                torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], idx
            ]
            + 1e-8
        )
        is_valid = mask[torch.arange(B)[:, None, None], idx]
        dist[~is_valid] = 0
        n_valid = is_valid.float().sum(dim=-1)
        return idx, (dist.sum(dim=-1) / (n_valid).clamp(min=1)).detach().clone()
    return idx


def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.
    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]

    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def region_mean(num_regions):
    """
    Input:
        num_regions - number of regions
    Return:
        means of regions
    """

    n = num_regions
    lookup = []
    d = 2 / n  # the cube size length
    #  construct all possibilities on the line [-1, 1] in the 3 axes
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            for k in range(n - 1, -1, -1):
                lookup.append([1 - d * (i + 0.5), 1 - d * (j + 0.5), 1 - d * (k + 0.5)])
    lookup = np.array(lookup)  # n**3 x 3
    return lookup


# def region_mean(num_regions=3, rng=3):
#     """
#     Input:
#         num_regions - number of regions
#     Return:
#         means of regions
#     """

#     n = num_regions
#     lookup = []
#     d = rng * 2 / n  # the cube size length
#     #  construct all possibilities on the line [-1, 1] in the 3 axes
#     for i in range(n - 1, -1, -1):
#         for j in range(n - 1, -1, -1):
#             for k in range(n-1, -1, -1):
#                 lookup.append([rng - d * (i + 0.5), rng - d * (j + 0.5), rng - d * (k + 0.5)])
#     lookup = np.array(lookup)  # n**3 x 3
#     return lookup


def assign_region_to_point(X, device="cuda:0", NREGIONS=3):
    """
    Input:
        X: point cloud [B, C, N]
        device: cuda:0, cpu
    Return:
        Y: Region assignment per point [B, N]
    """

    n = NREGIONS
    d = 2 / n
    X_clip = torch.clamp(X, -0.99999999, 0.99999999)  # [B, C, N]
    batch_size, _, num_points = X.shape
    Y = torch.zeros(
        (batch_size, num_points), device=device, dtype=torch.long
    )  # label matrix  [B, N]

    # The code below partitions all points in the shape to voxels.
    # At each iteration find per axis the lower threshold and the upper threshold values
    # of the range according to n (e.g., if n=3, then: -1, -1/3, 1/3, 1 - there are 3 ranges)
    # and save points in the corresponding voxel if they fall in the examined range for all axis.
    region_id = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0, :]  # [B, 1, N]
                x_axis_ut = X_clip[:, 0, :] < -1 + (x + 1) * d  # [B, 1, N]
                y_axis_lt = -1 + y * d < X_clip[:, 1, :]  # [B, 1, N]
                y_axis_ut = X_clip[:, 1, :] < -1 + (y + 1) * d  # [B, 1, N]
                z_axis_lt = -1 + z * d < X_clip[:, 2, :]  # [B, 1, N]
                z_axis_ut = X_clip[:, 2, :] < -1 + (z + 1) * d  # [B, 1, N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = torch.cat(
                    [x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut, z_axis_lt, z_axis_ut],
                    dim=1,
                ).view(
                    batch_size, 6, -1
                )  # [B, 6, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask, _ = torch.min(in_range, dim=1)  # [B, N]
                Y[mask] = region_id  # label each point with the region id
                region_id += 1

    return Y


# def assign_region_to_point(X, device, NREGIONS=3, rng=3):
#     """
#     Input:
#         X: point cloud [B,C,N]
#         device: cuda:0, cpu
#     Return:
#         Y: Region assignment per point [B, N]
#     """

#     n = NREGIONS
#     d = 2 * rng / n
#     X_clip = torch.clamp(X, -1*rng + 1e-8, rng - 1e-8) #-0.99999999, 0.99999999)  # [B, C, N]
#     Y = 0
#     for i in range(n-1):
#         Y = Y + n * n * ((X_clip[:, 0] > -rng + (i+1)*d).float())
#         Y = Y + n * ((X_clip[:, 1] > -rng + (i+1)*d).float())
#         Y = Y + ((X_clip[:, 2] > -rng + (i+1)*d).float())
#     return Y


def collapse_to_point(x, device):
    """
    Input:
        X: point cloud [C, N]
        device: cuda:0, cpu
    Return:
        x: A deformed point cloud. Randomly sample a point and cluster all point
        within a radius of RADIUS around it with some Gaussian noise.
        indices: the points that were clustered around x
    """
    # get pairwise distances
    inner = -2 * torch.matmul(x.transpose(1, 0), x)
    xx = torch.sum(x**2, dim=0, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(1, 0)

    # get mask of points in threshold
    mask = pairwise_distance.clone()
    mask[mask > RADIUS**2] = 100
    mask[mask <= RADIUS**2] = 1
    mask[mask == 100] = 0

    # Choose only from points that have more than MIN_POINTS within a RADIUS of them
    pts_pass = torch.sum(mask, dim=1)
    pts_pass[pts_pass < MIN_POINTS] = 0
    pts_pass[pts_pass >= MIN_POINTS] = 1
    indices = (pts_pass != 0).nonzero()

    # pick a point from the ones that passed the threshold
    point_ind = np.random.choice(indices.squeeze().cpu().numpy())
    point = x[:, point_ind]  # get point
    point_mask = mask[point_ind, :]  # get point mask

    # draw a gaussian centered at the point for points falling in the region
    indices = (point_mask != 0).nonzero().squeeze()
    x[:, indices] = torch.tensor(
        draw_from_gaussian(point.cpu().numpy(), len(indices)), dtype=torch.float
    ).to(device)
    return x, indices


def draw_from_gaussian(mean, num_points):
    """
    Input:
        mean: a numpy vector
        num_points: number of points to sample
    Return:
        points sampled around the mean with small std
    """
    return np.random.multivariate_normal(mean, np.eye(3) * 0.1, num_points).T  # 0.001


def draw_from_uniform(gap, region_mean, num_points):
    """
    Input:
        gap: a numpy vector of region x,y,z length in each direction from the mean
        region_mean:
        num_points: number of points to sample
    Return:
        points sampled uniformly in the region
    """
    return np.random.uniform(region_mean - gap, region_mean + gap, (num_points, 3)).T


def farthest_point_sample(xyz, npoint, device="cuda:0"):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # B x npoint
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(
            B, C, 1
        )  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum(
            (xyz - centroid) ** 2, 1
        )  # euclidean distance of points from the current centroid
        mask = (
            dist < distance
        )  # save index of all point that are closer than the current max distance
        distance[mask] = dist[
            mask
        ]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[
            1
        ]  # get the index of the point farthest away
    return centroids, centroids_vals


def farthest_point_sample_np(xyz, norm_curv=None, npoint=1024):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    centroids_norm_curv_vals = None
    if norm_curv is not None:
        centroids_norm_curv_vals = np.zeros((B, norm_curv.shape[1], npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(
            B, C, 1
        )  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        if norm_curv is not None:
            centroid_norm_curv = norm_curv[batch_indices, :, farthest].reshape(B, -1, 1)
            centroids_norm_curv_vals[:, :, i] = centroid_norm_curv[:, :, 0].copy()
        dist = np.sum(
            (xyz - centroid) ** 2, 1
        )  # euclidean distance of points from the current centroid
        mask = (
            dist < distance
        )  # save index of all point that are closer than the current max distance
        distance[mask] = dist[
            mask
        ]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(
            distance, axis=1
        )  # get the index of the point farthest away
    return centroids, centroids_vals, centroids_norm_curv_vals


def rotate_pc(pointcloud, reverse=False):
    if reverse:
        pointcloud = rotate_shape(pointcloud, "x", np.pi / 2)
    else:
        pointcloud = rotate_shape(pointcloud, "x", -np.pi / 2)
    return pointcloud


def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    R_y = np.asarray(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    R_z = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    if axis == "x":
        return x.dot(R_x).astype("float32")
    elif axis == "y":
        return x.dot(R_y).astype("float32")
    else:
        return x.dot(R_z).astype("float32")


def rotate_shape_tensor(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    if axis == "x":
        R_x = torch.tensor(
            [
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ]
            ]
        ).to(
            x
        )  # 1 x 3 x 3
        return x @ R_x
    elif axis == "y":
        R_y = torch.tensor(
            [
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            ]
        ).to(x)
        return x @ R_y
    else:
        R_z = torch.tensor(
            [
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            ]
        ).to(x)
        return x @ R_z


def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == "x":
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == "y":
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype("float32")


def random_rotate_one_axis_torch(X, axis="z"):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    # rotation_angle = np.random.uniform() * 2 * np.pi
    rotation_angle = torch.rand(len(X)).to(X.device) * 2 * np.pi
    cosval = torch.cos(rotation_angle)  # (batch_size,)
    sinval = torch.sin(rotation_angle)  # (batch_size,)
    ones = torch.ones_like(cosval)
    zeros = torch.zeros_like(cosval)
    if axis == "x":
        R_x = torch.stack(
            [
                torch.stack([ones, zeros, zeros], dim=-1),  # batch_size x 3
                torch.stack([zeros, cosval, -sinval], dim=-1),
                torch.stack([zeros, sinval, cosval], dim=-1),
            ],
            dim=1,
        )
        # R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = torch.matmul(X, R_x)
    elif axis == "y":
        R_y = torch.stack(
            [
                torch.stack([cosval, zeros, sinval], dim=-1),  # batch_size x 3
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sinval, zeros, cosval], dim=-1),
            ],
            dim=1,
        )
        # R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = torch.matmul(X, R_y)
    else:
        R_z = torch.stack(
            [
                torch.stack([cosval, -sinval, zeros], dim=-1),  # batch_size x 3
                torch.stack([sinval, cosval, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=1,
        )
        # R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = torch.matmul(X, R_z)
    return X.float()  # astype('float32')


def translate_pointcloud(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    # N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(*pointcloud.shape), -clip, clip)
    return pointcloud.astype("float32")


def jitter_pointcloud_adaptive(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape

    inner = np.matmul(pointcloud, np.transpose(pointcloud, (1, 0)))
    pc_2 = np.sum(pointcloud**2, axis=1, keepdims=True)
    pairwise_distances = pc_2 - 2 * inner + np.transpose(pc_2, (1, 0))
    zero_mask = np.where(pairwise_distances <= 1e-4)
    pairwise_distances[zero_mask] = 9999.0
    min_distances = np.min(pairwise_distances, axis=1)
    min_distances = np.sqrt(min_distances)

    min_distances_expdim = np.expand_dims(min_distances, axis=1)
    min_distances_expdim = np.repeat(min_distances_expdim, C, axis=1)

    # pointcloud += np.clip(min_distances_expdim * np.random.randn(N, C), -1 * min_distances_expdim, min_distances_expdim) # normal sampling
    pointcloud += np.clip(
        min_distances_expdim * (np.random.rand(N, C) * 2.0 - 1.0),
        -1 * min_distances_expdim,
        min_distances_expdim,
    )  # uniform sampling
    return pointcloud.astype("float32")


def scale(pc, scale_mode):
    pc = pc - pc.mean(0, keepdims=True)
    if scale_mode == "unit_std":
        pc /= np.std(pc)
    elif scale_mode == "unit_val":
        pc /= np.amax(np.abs(pc))
    elif scale_mode == "unit_norm":
        pc = scale_to_unit_cube(pc)
    else:
        raise ValueError("UNDEFINED SCALE MODE")
    return pc


def scale_to_unit_cube(x):
    """
    Input:
       x: pointcloud data, [B, C, N]
    Return:
       A point cloud scaled to unit cube
    """
    if len(x) == 0:
        return x
    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(x**2, axis=-1)))
    x /= furthest_distance
    return x


def scale_to_unit_cube_torch(x, only_mean=False, no_mean=False):
    """
    Input:
       x: pointcloud data, [B, N=1024, C=3]
    Return:
       A point cloud scaled to unit cube
    """
    assert len(x.shape) == 3
    if len(x) == 0:
        return x
    if not no_mean:
        centroid = torch.mean(x, dim=1, keepdim=True)
        x = x - centroid
    if not only_mean:
        furthest_distance = torch.max(
            torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True)), dim=1, keepdim=True
        ).values  # B x 1 x 1
        x = x / furthest_distance
    return x


def dropout_points(x, norm_curv, num_points):
    """
     Randomly dropout num_points, and randomly duplicate num_points
    Input:
        x: pointcloud data, [B, C, N]
    Return:
        A point cloud dropouted num_points
    """
    ind = random.sample(range(0, x.shape[1]), num_points)
    ind_dpl = random.sample(range(0, x.shape[1]), num_points)
    x[:, ind, :] = x[:, ind_dpl, :]
    norm_curv[:, ind, :] = norm_curv[:, ind_dpl, :]
    return x, norm_curv


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


def remove(points, p_keep=0.7):
    rand_xyz = uniform_2_sphere()
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid

    dist_from_plane = np.dot(points_centered, rand_xyz)
    if p_keep == 0.5:
        mask = dist_from_plane > 0
    else:
        mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

    return points[mask, :]


def remove_region_points(x, norm_curv, device):
    """
    Remove all points of a randomly selected region in the point cloud.
    Input:
        X - Point cloud [B, N, C]
        norm_curv: norm and curvature, [B, N, C]
    Return:
        X - Point cloud where points in a certain region are removed
    """
    # get points' regions
    regions = assign_region_to_point(x, device)  # [B, N] N:the number of region_id
    n = NREGIONS
    region_ids = np.random.permutation(n**3)
    for b in range(x.shape[0]):
        for i in region_ids:
            ind = regions[b, :] == i  # [N]
            # if there are enough points in the region
            if torch.sum(ind) >= 50:
                num_points = int(torch.sum(ind))
                rnd_ind = random.sample(range(0, x.shape[1]), num_points)
                x[b, ind, :] = x[b, rnd_ind, :]
                norm_curv[b, ind, :] = norm_curv[b, rnd_ind, :]
                break  # move to the next shape in the batch
    return x, norm_curv


def extract_feature_points(x, norm_curv, num_points, device="cuda:0"):
    """
    Input:
        x: pointcloud data, [B, N, C]
        norm_curv: norm and curvature, [B, N, C]
    Return:
        Feature points, [B, num_points, C]
    """
    IND = torch.zeros([x.size(0), num_points]).to(device)
    fea_pc = torch.zeros([x.size(0), num_points, x.size(2)]).to(device)
    for b in range(x.size(0)):
        curv = norm_curv[b, :, -1]
        curv = abs(curv)
        ind = torch.argsort(curv)
        ind = ind[:num_points]
        IND[b] = ind
        fea_pc[b] = x[b, ind, :]
    return fea_pc


def pc2voxel(x):
    # Args:
    #     x: size n x F where n is the number of points and F is feature size
    # Returns:
    #     voxel: N x N x N x (K x 3 + 1)
    #     index: N x N x N x K
    num_points = x.shape[0]
    data = np.zeros((N, N, N, NUM_FEATURES), dtype=np.float32)
    index = np.zeros((N, N, N, K), dtype=np.float32)
    x /= 1.05
    idx = np.floor((x + 1.0) / 2.0 * N)
    L = [[] for _ in range(N * N * N)]
    for p in range(num_points):
        k = int(idx[p, 0] * N * N + idx[p, 1] * N + idx[p, 2])
        L[k].append(p)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                u = int(i * N * N + j * N + k)
                if not L[u]:
                    data[i, j, k, :] = np.zeros((NUM_FEATURES), dtype=np.float32)
                elif len(L[u]) >= K:
                    choice = np.random.choice(L[u], size=K, replace=False)
                    local_points = x[choice, :] - np.array(
                        [
                            -1.0 + (i + 0.5) * 2.0 / N,
                            -1.0 + (j + 0.5) * 2.0 / N,
                            -1.0 + (k + 0.5) * 2.0 / N,
                        ],
                        dtype=np.float32,
                    )
                    data[i, j, k, 0 : K * 3] = np.reshape(local_points, (K * 3))
                    data[i, j, k, K * 3] = 1.0
                    index[i, j, k, :] = choice
                else:
                    choice = np.random.choice(L[u], size=K, replace=True)
                    local_points = x[choice, :] - np.array(
                        [
                            -1.0 + (i + 0.5) * 2.0 / N,
                            -1.0 + (j + 0.5) * 2.0 / N,
                            -1.0 + (k + 0.5) * 2.0 / N,
                        ],
                        dtype=np.float32,
                    )
                    data[i, j, k, 0 : K * 3] = np.reshape(local_points, (K * 3))
                    data[i, j, k, K * 3] = 1.0
                    index[i, j, k, :] = choice
    return data, index


def pc2voxel_B(x):
    """
    Input:
        x: pointcloud data, [B, num_points, C]
    Return:
        voxel: N x N x N x (K x 3 + 1)
        index: N x N x N x K
    """
    batch_size = x.shape[0]
    Data = np.zeros((batch_size, N, N, N, NUM_FEATURES), dtype=np.float32)
    Index = np.zeros((batch_size, N, N, N, K), dtype=np.float32)
    x = scale_to_unit_cube(x)
    for b in range(batch_size):
        pc = x[b]
        data, index = pc2voxel(pc)
        Data[b] = data
        Index[b] = index
    return Data, Index


def pc2image(X, axis, RESOLUTION=32):
    """
    Input:
        X: point cloud [N, C]
        axis: axis to do projection about
    Return:
        Y: image projected by 'X' along 'axis'. [32, 32]
    """

    n = RESOLUTION
    d = 2 / n
    X_clip = np.clip(X, -0.99999999, 0.99999999)  # [N, C]
    Y = np.zeros((n, n), dtype=np.float32)  # label matrix  [n, n]
    if axis == "x":
        for y in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                y_axis_lt = -1 + y * d < X_clip[:, 1]  # [N]
                y_axis_ut = X_clip[:, 1] < -1 + (y + 1) * d  # [N]
                z_axis_lt = -1 + z * d < X_clip[:, 2]  # [N]
                z_axis_ut = X_clip[:, 2] < -1 + (z + 1) * d  # [N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = np.concatenate(
                    [y_axis_lt, y_axis_ut, z_axis_lt, z_axis_ut], 0
                ).reshape(
                    4, -1
                )  # [4, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask = np.min(in_range, 0)  # [N]: [False, ..., True, ...]
                if np.sum(mask) == 0:
                    continue
                Y[y, z] = (X_clip[mask, 0] + 1).mean()
    if axis == "y":
        for x in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0]  # [N]
                x_axis_ut = X_clip[:, 0] < -1 + (x + 1) * d  # [N]
                z_axis_lt = -1 + z * d < X_clip[:, 2]  # [N]
                z_axis_ut = X_clip[:, 2] < -1 + (z + 1) * d  # [N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = np.concatenate(
                    [x_axis_lt, x_axis_ut, z_axis_lt, z_axis_ut], 0
                ).reshape(
                    4, -1
                )  # [4, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask = np.min(in_range, 0)  # [N]
                if np.sum(mask) == 0:
                    continue
                Y[x, z] = (X_clip[mask, 1] + 1).mean()
    if axis == "z":
        for x in range(n):
            for y in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0]  # [N]
                x_axis_ut = X_clip[:, 0] < -1 + (x + 1) * d  # [N]
                y_axis_lt = -1 + y * d < X_clip[:, 1]  # [N]
                y_axis_ut = X_clip[:, 1] < -1 + (y + 1) * d  # [N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = np.concatenate(
                    [x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut], 0
                ).reshape(
                    4, -1
                )  # [4, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask = np.min(in_range, 0)  # [N]
                if np.sum(mask) == 0:
                    continue
                Y[x, y] = (X_clip[mask, 2] + 1).mean()

    return Y


def pc2image_B(X, axis, device="cuda:0", RESOLUTION=32):
    """
    Input:
        X: point cloud [B, C, N]
        axis: axis to do projection about
    Return:
        Y: image projected by 'X' along 'axis'. [B, 32, 32]
    """
    n = RESOLUTION
    B = X.size(0)
    X = X.permute(0, 2, 1)  # [B, N, C]
    X = X.cpu().numpy()
    Y = np.zeros((B, n, n), dtype=np.float32)  # label matrix  [B, n, n]
    for b in range(B):
        Y[b] = pc2image(X[b], axis, n)
    Y = torch.from_numpy(Y).to(device)
    return Y


def CPL(x, ratio):
    """
    Input:
        x: points feature [N, C]
        ratio: down sampling ratio
    Return:
        f_out: down sampled points feature, [M, C]
    """
    num_sample = int(np.size(x, 0) / ratio)
    fs = np.array([])
    fr = np.array([]).astype(int)
    fmax = x.max(0)
    idx = x.argmax(0)
    _, d = np.unique(idx, return_index=True)
    uidx = np.argsort(d)
    for i in uidx:
        mask = i == idx
        val = fmax[mask].sum()
        fs = np.append(fs, val)
        fr = np.append(fr, mask.sum())
    sidx = np.argsort(-fs)
    suidx = uidx[sidx]
    fr = fr[sidx]
    midx = np.array([]).astype(int)
    t = 0
    for i in fr:
        for j in range(int(i)):
            midx = np.append(midx, suidx[t])
        t += 1
    rmidx = np.resize(midx, num_sample)
    fout = x[rmidx]
    return fout


def CPL_B(
    X,
    ratio,
    device="cuda:0",
):
    """
    Input:
        X: points feature [B, C, N]
        ratio: down sampling ratio
    Return:
        F: down sampled points feature, [B, C, M]
    """
    B, C, N = X.size()
    M = int(N / ratio)
    X = X.permute(0, 2, 1)  # [B, N, C]
    X = X.cpu().numpy()
    F = np.zeros((B, M, C), dtype=np.float32)
    for b in range(B):
        F[b] = CPL(X[b], ratio)
    F = torch.from_numpy(F).to(device)
    F = F.permute(0, 2, 1)
    return F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_ball_group(s, radius, n, coords, features):
    """
    Sampling by FPS and grouping by ball query.
    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by ball query
        n[int]: fix number of points in ball neighbor
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]

    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx, _ = farthest_point_sample(coords, s)  # [B, s]
    new_coords = index_points(coords, fps_idx)  # [B, s, 3]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # ball_query grouping
    idx = query_ball_point(radius, n, coords, new_coords)  # [B, s, n]
    grouped_features = index_points(features, idx)  # [B, s, n, D]

    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(
        batch_size, s, 1, -1
    )  # [B, s, n, D]

    # Concat, my be different in many networks
    aggregated_features = torch.cat(
        [
            grouped_features_norm,
            new_features.view(batch_size, s, 1, -1).repeat(1, 1, n, 1),
        ],
        dim=-1,
    )  # [B, s, n, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, n, 2D]


def sample_and_knn_group(k, features, lg, hard=False):
    """
    Sampling by gumbel_softmax and grouping by KNN.
    Input:
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
        lg[tensor]: logits data with size of [B, N, S]
        hard[bool]: gumbel sampling with soft or hard

    Returns:
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size, s = lg.shape[0], lg.shape[1]

    # gumbel sampling
    # y = gumbel_softmax(lg, 0.2, hard)  # y:[B, s, N]
    # new_features = torch.bmm(y, features)

    fps_idx, _ = farthest_point_sample(features, s)  # [B, s]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, features, new_features)  # [B, s, k]
    grouped_features = index_points(features, idx)  # [B, s, k, D]

    # Matrix subtraction
    grouped_features_norm = grouped_features - new_features.view(
        batch_size, s, 1, -1
    )  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat(
        [
            grouped_features_norm,
            new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1),
        ],
        dim=-1,
    )  # [B, s, k, 2D]

    return aggregated_features  # [B, s, k, 2D]


def defcls_input(
    X, norm_curv=None, lookup=None, device="cuda:0", NREGIONS=3, rng=4, drop_rate=0.5
):
    """
    Deform a region in the point cloud.
    Input:
        args - commmand line arguments
        X - Point cloud [B, N, C]
        norm_curv - norm and curvature [B, N, D]
        lookup - regions center point
        device - cuda/cpu
    Return:
        X - Point cloud with a deformed region
        def_label - {0,1,...,26} indicating the deform class (deform region location) respectively
    """

    # get points' regions
    n = NREGIONS
    regions = assign_region_to_point(X, device, NREGIONS, rng=rng)  # [B, N]
    drop_prob = X.new_zeros((regions.shape[0], n * n * n))
    drop_prob.scatter_add_(1, regions.long(), torch.ones_like(regions))

    drop_region = torch.bernoulli(
        (drop_prob > 10).bool().float() * drop_rate,
        # drop_rate*X.new_ones((regions.shape[0], NREGIONS*NREGIONS*NREGIONS)),
    ).bool()  # [B, NREGIONS*NREGIONS*NREGIONS]  1=drop 0=keep
    while torch.any(
        ((drop_prob > 10).float() * (~drop_region).float()).sum(dim=1) == 0
    ):
        drop_region = torch.bernoulli(
            (drop_prob > 0).bool().float() * drop_rate,
            # drop_rate*X.new_ones((regions.shape[0], NREGIONS*NREGIONS*NREGIONS)),
        ).bool()  # [B, NREGIONS*NREGIONS*NREGIONS]  1=drop 0=keep

    region = torch.arange(n * n * n).unsqueeze(0).expand_as(drop_region).to(device)
    region[~drop_region] = -1
    dropped = (regions.unsqueeze(2) == region.unsqueeze(1)).float().sum(
        -1
    ) > 0  # [B, N, nxn] -> [B, N]
    mask = (~dropped).float().unsqueeze(2)  # B x N x 1
    mask_bool = (~dropped).unsqueeze(2)  # B x N x 1
    assert torch.all(mask.sum(dim=1) > 10)

    sum = (X * mask).sum(dim=1, keepdim=True)  # B x 1 x 3
    num = mask.sum(dim=1, keepdim=True)  # B x 1 x 1
    mean = sum / num  # B x 1 x 3
    X = (X - mean) * mask
    # unit_std
    std = (
        X.pow(2).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) / (num * 3)
    ).sqrt()  # B x 1 x 1
    X = X / std  # * mask #(~dropped).float().unsqueeze(2)

    while torch.any(mask == 0):
        sorted_mask, order = torch.sort(mask.squeeze(-1), dim=1, descending=True)
        empty = (1 - sorted_mask).sum(dim=1)

    return X, mask, drop_region, mean, std


def random_radius_drop(X, radius):
    max_norm = X.norm(dim=2).max(dim=1).values


def create_folders(args):
    try:
        os.makedirs("outputs")
    except OSError:
        pass
    try:
        os.makedirs("outputs/" + args.exp_name)
    except OSError:
        pass


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path), map_location="cpu")
    model.eval()
    return model


# Gradient clipping
class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0
    )

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(
            f"Clipped gradient with value {grad_norm:.1f} "
            f"while allowed {max_grad_norm:.1f}"
        )
    return grad_norm


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = -sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        # x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        # x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        # x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")
    return x.contiguous()


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device("cpu"))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(
            torch.device("cuda:%d" % gpu)
        )
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]

    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
    )  # batch*3
    return out


def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def softmax(x):
    max = np.max(
        x, axis=-1, keepdims=True
    )  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(
        e_x, axis=-1, keepdims=True
    )  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


# TODO: implement mean/median filtering
def low_pass_filtering(x, method, scale):
    if not method or scale == 1:
        return x
    elif method == "fps":
        return farthest_point_sample(
            x.swapaxes(1, 2), npoint=int(x.shape[1] / scale), device=x.device
        )[-1].swapaxes(1, 2)
    elif method == "mean":
        return x
    elif method == "median":
        return x


def get_color(
    coords,
    corners=np.array(
        [
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ]
    )
    * 3,
    mask=None,
):  # Visualization
    coords = np.array(coords)  # batch x n_points x 3
    corners = np.array(corners)  # n_corners x 3
    colors = np.array(
        [
            [255, 0, 0],
            [255, 127, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [75, 0, 130],
            [143, 0, 255],
        ]
    )

    dist = np.linalg.norm(coords[:, :, None, :] - corners[None, None, :, :], axis=-1)
    weight = softmax(-dist)[:, :, :, None]  # batch x NUM_POINTS x n_corners x 1
    rgb = (weight * colors).sum(2).astype(int)  # NUM_POINTS x 3
    if mask is not None:
        # mask : B x N
        # rgb : B x N x 3 (RGB)
        rgb[(~mask).nonzero()] = np.array([255, 255, 255])
    return rgb


if __name__ == "__main__":
    lookup = region_mean(3)
    print(lookup.shape)
    x = np.random.rand(2, 3, 6)  # [B, C, N]
    print(x)
    x = scale_to_unit_cube(x)
    x = torch.from_numpy(x)
    print(x)
    # dropout_points(x, 2)
    y = pc2image_B(x, "x", RESOLUTION=6)
    print(y.shape)
    x = torch.stack(
        (
            pc2image_B(x, "x", RESOLUTION=6),
            pc2image_B(x, "y", RESOLUTION=6),
            pc2image_B(x, "z", RESOLUTION=6),
        ),
        dim=3,
    )
    print(x)
