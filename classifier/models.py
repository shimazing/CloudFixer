import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pc_utils import *



class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

    def get_feature(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = self.bn6(self.linear1(x))
        return x

    def get_low_level_feature(self, x):
        x = self.bn1(self.conv1(x))
        return x



class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, return_feature=False):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        if self.leaky_relu:
            act = lambda y: F.leaky_relu(y, negative_slope=0.2)
        else:
            act = F.relu

        x = act(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        if return_feature:
            return self.bn7(self.linear2(x))


        x = act(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

    def get_feature(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        if self.leaky_relu:
            act = lambda y: F.leaky_relu(y, negative_slope=0.2)
        else:
            act = F.relu

        x = act(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.bn7(self.linear2(x))
        return x

    def get_low_level_feature(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        return x1



class DGCNNWrapper(nn.Module):
    def __init__(self, dataset, output_channels):
        super().__init__()
        class Args:
            def __init__(self):
                self.k = 20
                self.emb_dims = 1024
                self.dropout = 0.5
                self.leaky_relu = 1
        args = Args()
        self.dataset = dataset
        self.model = DGCNN(args, output_channels=output_channels)

    def forward(self, pc, return_feature=False):
        # normalize pc here!
        pc = scale_to_unit_cube_torch(pc)
        if self.dataset.startswith("modelnet40"):
            pc = rotate_shape_tensor(pc, 'x', np.pi/2)
        pc = pc.permute(0, 2, 1).contiguous()
        logit = self.model(pc, return_feature)
        return logit

    def get_feature(self, pc):
        pc = scale_to_unit_cube_torch(pc)
        if self.dataset.startswith("modelnet40"):
            pc = rotate_shape_tensor(pc, 'x', np.pi/2)
        pc = pc.permute(0, 2, 1).contiguous()
        return self.model.get_feature(pc)


    def get_low_level_feature(self, pc):
        pc = scale_to_unit_cube_torch(pc)
        if self.dataset.startswith("modelnet40"):
            pc = rotate_shape_tensor(pc, 'x', np.pi/2)
        pc = pc.permute(0, 2, 1).contiguous()
        return self.model.get_low_level_feature(pc)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def min_max_normalize(pc):
    # pc: B x N (=1024) x C (=3)
    assert pc.shape[2] == 3
    new_pc = pc.detach().clone()
    new_pc -= (pc.min(dim=1, keepdim=True).values + pc.max(dim=1, keepdim=True).values) / 2
    length = (pc.max(dim=1, keepdim=True).values - pc.min(dim=1,keepdim=True).values).max(dim=2, keepdim=True).values
    new_pc *= 2.0 / length
    return new_pc