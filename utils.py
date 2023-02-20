import numpy as np
import getpass
import os
import torch

def random_rotate_one_axis_torch(X, axis='z'):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    #rotation_angle = np.random.uniform() * 2 * np.pi
    rotation_angle = torch.rand(len(X)).to(X.device) * 2 * np.pi
    cosval = torch.cos(rotation_angle) # (batch_size,)
    sinval = torch.sin(rotation_angle) # (batch_size,)
    ones = torch.ones_like(cosval)
    zeros = torch.zeros_like(cosval)
    if axis == 'x':
        R_x = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1), # batch_size x 3
            torch.stack([zeros, cosval, -sinval], dim=-1),
            torch.stack([zeros, sinval, cosval], dim=-1)], dim=1)
        #R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = torch.matmul(X, R_x)
    elif axis == 'y':
        R_y = torch.stack([
            torch.stack([cosval, zeros, sinval], dim=-1), # batch_size x 3
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sinval, zeros, cosval], dim=-1)], dim=1)
        #R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = torch.matmul(X, R_y)
    else:
        R_z = torch.stack([
            torch.stack([cosval, -sinval, zeros], dim=-1), # batch_size x 3
            torch.stack([sinval, cosval, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)], dim=1)
        #R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = torch.matmul(X, R_z)
    return X.float() #astype('float32')


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
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')


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

def region_mean(num_regions=3, rng=4):
    """
    Input:
        num_regions - number of regions
    Return:
        means of regions
    """

    n = num_regions
    lookup = []
    d = rng * 2 / n  # the cube size length
    #  construct all possibilities on the line [-1, 1] in the 3 axes
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            for k in range(n-1, -1, -1):
                lookup.append([rng - d * (i + 0.5), rng - d * (j + 0.5), rng - d * (k + 0.5)])
    lookup = np.array(lookup)  # n**3 x 3
    return lookup

def assign_region_to_point(X, device, NREGIONS=3, rng=4):
    """
    Input:
        X: point cloud [B,N,C]
        device: cuda:0, cpu
    Return:
        Y: Region assignment per point [B, N]
    """

    n = NREGIONS
    d = 2 * rng / n
    X_clip = torch.clamp(X, -1*rng + 1e-8, rng - 1e-8).permute(0,2,1) #-0.99999999, 0.99999999)  # [B, C, N]

    Y = n * n * ((X_clip[:, 0] > -rng + d).float() + (X_clip[:, 0] >
        -rng+2*d).long()) # x
    Y += n * ((X_clip[:, 1] > -rng + d).float() + (X_clip[:, 1] >
        -rng+2*d).long()) # y
    Y += ((X_clip[:, 2] > -rng + d).long() + (X_clip[:, 2] > -rng+2*d).float()) # z

    return Y
    #batch_size, _, num_points = X.shape
    #Y = torch.zeros((batch_size, num_points), device=device, dtype=torch.long)  # label matrix  [B, N]

    ## The code below partitions all points in the shape to voxels.
    ## At each iteration find per axis the lower threshold and the upper threshold values
    ## of the range according to n (e.g., if n=3, then: -1, -1/3, 1/3, 1 - there are 3 ranges)
    ## and save points in the corresponding voxel if they fall in the examined range for all axis.

    #region_id = 0
    #for x in range(n):
    #    for y in range(n):
    #        for z in range(n):
    #            # lt= lower threshold, ut = upper threshold
    #            x_axis_lt = -1*rng + x * d < X_clip[:, 0, :]  # [B, 1, N]
    #            x_axis_ut = X_clip[:, 0, :] < -1*rng + (x + 1) * d  # [B, 1, N]
    #            y_axis_lt = -1*rng + y * d < X_clip[:, 1, :]  # [B, 1, N]
    #            y_axis_ut = X_clip[:, 1, :] < -1*rng + (y + 1) * d  # [B, 1, N]
    #            z_axis_lt = -1*rng + z * d < X_clip[:, 2, :]  # [B, 1, N]
    #            z_axis_ut = X_clip[:, 2, :] < -1*rng + (z + 1) * d  # [B, 1, N]
    #            # get a mask indicating for each coordinate of each point of each shape whether
    #            # it falls inside the current inspected ranges
    #            in_range = torch.cat([x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut,
    #                                  z_axis_lt, z_axis_ut], dim=1).view(batch_size, 6, -1)  # [B, 6, N]
    #            # per each point decide if it falls in the current region only if in all
    #            # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
    #            mask, _ = torch.min(in_range, dim=1)  # [B, N]
    #            Y[mask] = region_id  # label each point with the region id
    #            region_id += 1
    #return Y

def defcls_input(X, norm_curv=None, lookup=None, device='cuda:0', NREGIONS=3, rng=4,
        drop_rate=0.5):
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
    drop_prob = X.new_zeros((regions.shape[0], n*n*n))
    drop_prob.scatter_add_(1, regions.long(), torch.ones_like(regions))

    drop_region = torch.bernoulli(
            (drop_prob > 10).bool().float() * drop_rate,
            #drop_rate*X.new_ones((regions.shape[0], NREGIONS*NREGIONS*NREGIONS)),
            ).bool() # [B, NREGIONS*NREGIONS*NREGIONS]  1=drop 0=keep
    while torch.any(((drop_prob > 10).float() * (~drop_region).float()).sum(dim=1) == 0):
        drop_region = torch.bernoulli(
                (drop_prob > 0).bool().float() * drop_rate,
                #drop_rate*X.new_ones((regions.shape[0], NREGIONS*NREGIONS*NREGIONS)),
                ).bool() # [B, NREGIONS*NREGIONS*NREGIONS]  1=drop 0=keep

    region = torch.arange(n*n*n).unsqueeze(0).expand_as(drop_region).to(device)
    region[~drop_region] = -1
    dropped = (regions.unsqueeze(2) == region.unsqueeze(1)).float().sum(-1) > 0 # [B, N, nxn] -> [B, N]
    mask = (~dropped).float().unsqueeze(2) # B x N x 1
    assert torch.all(mask.sum(dim=1) > 10)

    sum = (X * mask).sum(dim=1, keepdim=True) # B x 1 x 3
    num = mask.sum(dim=1, keepdim=True) # B x 1 x 1
    mean = sum / num # B x 1 x 3
    X = (X - mean) * mask
    # unit_std
    std = (X.pow(2).sum(dim=2, keepdim=True).sum(dim=1,
            keepdim=True) / (num * 3)).sqrt() # B x 1 x 1
    X = X / std * mask #(~dropped).float().unsqueeze(2)
    return X, mask, drop_region, mean, std


# 예측을 기반으로 다시 PC를 원래대로 돌리려면 ?
# dropped 된 곳에 noise를 생성하고 이것으로 zero centering을 위해 옮겨주기?
# mean, std 예측한걸로 기존 point를 옮겨주고, drop region으로 예측된 부분에
# point를 추가


# drop된 곳 기반으로 빠진 mean 계산하
# 아니면 같은 network로 기존의 mean, std 도 예측해주기로..!
# 그 다음 mean, std, prediction 한 drop region 을 =


    #curv_conf = torch.ones(X.shape[0]).to(device)
    #region_ids = np.random.permutation(n ** 3)
    #region_ids = torch.from_numpy(region_ids).to(device)
    #def_label = torch.zeros(regions.size(0)).long().to(device)  # binary mask of deformed points

    #for b in range(X.shape[0]):
    #    for i in region_ids:
    #        ind = regions[b, :] == i  # [N]
    #        # if there are enough points in the region
    #        if torch.sum(ind) >= pc_utils.MIN_POINTS:
    #            region = lookup[i].cpu().numpy()  # current region average point
    #            def_label[b] = i
    #            num_points = int(torch.sum(ind).cpu().numpy())
    #            rnd_pts = pc_utils.draw_from_gaussian(region, num_points)  # generate region deformation points
    #            # rnd_ind = random.sample(range(0, X.shape[2]), num_points)
    #            # X[b, :, ind] = X[b, :, rnd_ind]
    #            curv_conf[b] = norm_curv[b, ind, -1].abs().sum() / norm_curv[b, :, -1].abs().sum()
    #            X[b, :3, ind] = torch.tensor(rnd_pts, dtype=torch.float).to(device)  # replace with region deformation points
    #            break  # move to the next shape in the batch

    #return X, def_label, curv_conf

def random_radius_drop(X, radius):
    max_norm = X.norm(dim=2).max(dim=1).values



# Folders
def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass


# Model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#Gradient clipping
class Queue():
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
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm


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
        Rx[:, 2:3, 1:2] = - sin
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
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()


# Other utilities
def get_wandb_username(username):
    if username == 'cvignac':
        return 'cvignac'
    current_user = getpass.getuser()
    if current_user == 'victor' or current_user == 'garciasa':
        return 'vgsatorras'
    else:
        return username


if __name__ == "__main__":


    ## Test random_rotation
    bs = 2
    n_nodes = 16
    n_dims = 3
    x = torch.randn(bs, n_nodes, n_dims)
    print(x)
    x = random_rotation(x)
    #print(x)
