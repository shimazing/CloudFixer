import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from utils_GAST.trans_norm import TransNorm2d
except:
    from utils.trans_norm import TransNorm2d
from torch.autograd import Function

from torch.nn.modules.batchnorm import BatchNorm2d
import numpy as np

class GroupNormTimeCond(torch.nn.GroupNorm):
    def __init__(self, num_channels=64, num_groups=32, t_dim=512):
        super().__init__(num_groups=num_groups, num_channels=num_channels)
        self.t_proj = nn.Linear(t_dim, 2*num_channels)

    def forward(self, input, t_emb):
        t_emb = self.t_proj(t_emb)
        weight, bias = t_emb.chunk(2, dim=1)

        output = super().forward(input)
        output = output * (weight[:, :, None, None] + 1) + bias[:,:, None, None]
        return output




class BatchNorm2dTimeCond(BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        t_dim=512,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNorm2dTimeCond, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.t_proj = nn.Linear(t_dim, 2*num_features)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input, t_emb):
        t_emb = self.t_proj(t_emb)
        weight, bias = t_emb.chunk(2, dim=1)
        #delattr(self, "weight")
        #delattr(self, "bias")
        #self.weight = weight[:, :, None, None]
        #self.bias = bias[:, :, None, None]

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        output = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        return output * weight[:, :, None, None] + bias[:, :, None, None]


K = 20

def calc_t_emb(ts, t_emb_dim=128):
    """
    Embed time steps into a higher dimension space
    """
    assert t_emb_dim % 2 == 0

    # input is of shape (B) of integer time steps
    # output is of shape (B, t_emb_dim)
    ts = ts.unsqueeze(1)
    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.to(ts.device) # shape (half_dim)
    # ts is of shape (B,1)
    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)

    return t_emb


def knn(x, k, mask=None):
    # mask : [B, N]
    # x : [B, C=3, N]
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  #거리 가장 가까운거 골라야하니까 음수 붙여줌
    if mask is not None:
        B_ind, N_ind = (~mask).nonzero(as_tuple=True)
        pairwise_distance[B_ind, N_ind] = np.inf
        pairwise_distance[B_ind, :, N_ind] = np.inf
    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, args, k=20, idx=None, mask=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k, mask=mask)  # (batch_size, num_points, k)
    # Run on cpu or gpu
    device = torch.device("cuda:" + str(x.get_device()) if args.cuda else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    # batch_size x feature_dim x num_points x k
    feature_mask = None
    if mask is not None:
        masked_num_points = mask.float().sum(dim=1).to(device)
        feature_mask = (torch.arange(k).view(1, -1).to(device) < masked_num_points.view(-1,
            1)).view(batch_size, 1, 1, k).float()
        feature = feature * feature_mask

    return feature, feature_mask


def l2_norm(input, axit=1):
    norm = torch.norm(input, 2, axit, True)
    output = torch.div(input, norm)
    return output


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                # nn.InstanceNorm2d(out_ch),
                # TransNorm2d(out_ch),
                #nn.LayerNorm([out_ch, 1024, 20]),
                nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                # nn.InstanceNorm2d(out_ch),
                # TransNorm2d(out_ch),
                #nn.LayerNorm([out_ch, 1024, 20]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_2d_time_cond(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, t_dim=512, activation='relu', bias=True):
        super(conv_2d_time_cond, self).__init__()
        #self.time_proj = nn.Linear(t_dim, out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias)
        #self.bn = BatchNorm2dTimeCond(out_ch, t_dim=t_dim)
        self.bn = GroupNormTimeCond(num_channels=out_ch, t_dim=t_dim)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, t_emb):
        #t_emb = self.time_proj(t_emb)
        x = self.conv(x)
        x = self.bn(x, t_emb)
        x = self.act(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, activation='relu', bias=True,
            norm=True):
        super(fc_layer, self).__init__()
        self.norm = norm
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                #nn.BatchNorm1d(out_ch),
                nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        if self.norm:
            x = l2_norm(x, 1)
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """

    def __init__(self, args, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out
        self.args = args

        activation = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = False if args.model == 'dgcnn' else True

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation=activation, bias=bias)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation=activation, bias=bias)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation=activation, bias=bias)
        self.fc1 = fc_layer(1024, 512, activation=activation, bias=bias,
                bn=True,
            norm=getattr(args, 'fc_norm', True)
                )
        self.fc2 = fc_layer(512, 256, activation=activation, bn=True,
            norm=getattr(args, 'fc_norm', True)
                )
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        device = torch.device("cuda:" + str(x.get_device()) if self.args.cuda else "cpu")

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        if self.args.model == "dgcnn":
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


class PointNet(nn.Module):
    def __init__(self, args, num_class=10):
        super(PointNet, self).__init__()
        self.args = args

        self.trans_net1 = transform_net(args, 3, 3)
        self.trans_net2 = transform_net(args, 64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        num_f_prev = 64 + 64 + 64 + 128

        self.cls_C = class_classifier(args, 1024, 10)
        self.domain_C = domain_classifier(args, 1024, 2)
        self.rotcls_C1 = linear_classifier(1024, 4)
        self.rotcls_C2 = linear_classifier(1024, 4)
        self.defcls_C = ssl_classifier(args, 1024, getattr(args, 'nregions', 3)**3)
        self.DecoderFC = DecoderFC(args, 1024)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)
        self.normreg_C = nn.Conv1d(1024, 4, kernel_size=1, bias=False)

    def forward(self, x, alpha=0, activate_DefRec=False):
        num_points = x.size(2)
        x = torch.unsqueeze(x, dim=3)

        cls_logits = {}

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        transform = self.trans_net2(x2)
        x = x2.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x5 = self.conv5(x4)
        x5_pool, _ = torch.max(x5, dim=2, keepdim=False)
        x = x5_pool.squeeze(dim=2)  # batchsize*1024

        cls_logits["cls"] = self.cls_C(x)
        if alpha != 0:
            reverse_x = ReverseLayerF.apply(x, alpha)
            cls_logits["domain_cls"] = self.domain_C(reverse_x)
        cls_logits["rot_cls1"] = self.rotcls_C1(x)
        cls_logits["rot_cls2"] = self.rotcls_C2(x)
        cls_logits["def_cls"] = self.defcls_C(x)
        # cls_logits["curv_conf"] = self.curvconfreg_C(x)
        # cls_logits["norm_reg"] = self.normreg_C(x5).permute(0, 2, 1)
        cls_logits["decoder"] = self.DecoderFC(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat.squeeze(dim=3), x5_pool.repeat(1, 1, num_points)), dim=1)
            cls_logits["DefRec"] = self.DefRec(DefRec_input)

        return cls_logits

def swish(x):
    return x * torch.sigmoid(x)

class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args

        self.time_cond = getattr(args, 'time_cond', False)
        if getattr(args, 'time_cond', False):
            t_dim = 128
            self.t_dim = t_dim
            self.fc_t1 = nn.Linear(t_dim, 4*t_dim)
            self.fc_t2 = nn.Linear(4*t_dim, 4*t_dim)
            self.activation = swish

        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)
        if not self.time_cond:
            self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
            self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
            self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
            self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        else:
            self.conv1 = conv_2d_time_cond(6, 64, kernel=1, bias=False, activation='leakyrelu')
            self.conv2 = conv_2d_time_cond(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
            self.conv3 = conv_2d_time_cond(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
            self.conv4 = conv_2d_time_cond(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        if self.time_cond:
            self.bn5 = nn.LayerNorm([512, 1024]) # TODO for GN mode
        else:
            self.bn5 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(num_f_prev, 512, kernel_size=1, bias=False)

        self.cls_C = class_classifier(args, 1024, 10)
        self.domain_C = domain_classifier(args, 1024, 2)
        self.rotcls_C1 = linear_classifier(1024, 4)
        self.rotcls_C2 = linear_classifier(1024, 4)
        self.defcls_C = linear_classifier(1024, getattr(args, 'nregions', 3)**3)
        if getattr(args, 'pred_stat', False):
            self.stat_C = linear_classifier(1024, getattr(args, 'nregions',
                3)*3 + 1)
        if getattr(args, 'cl', False):
            self.cl_head = ssl_classifier(args, 1024, args.cl_dim)
        # self.normreg_C = nn.Conv1d(1024, 4, kernel_size=1, bias=False)
        # self.curvconfreg_C = linear_classifier(1)
        self.DecoderFC = DecoderFC(args, 1024)

        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, alpha=0, ts=None, activate_DefRec=False, mask=None):
        if self.time_cond:
            assert ts is not None
            t_emb = calc_t_emb(ts, self.t_dim)
            t_emb = self.fc_t1(t_emb)
            t_emb = self.activation(t_emb)
            t_emb = self.fc_t2(t_emb)
            t_emb = self.activation(t_emb)

        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        if getattr(self.args, 'input_transform', False):
            x0, _ = get_graph_feature(x, self.args, k=self.k, mask=mask)  # x0: [b, 6, 1024, 20]
            # align to a canonical space (e.g., apply rotation such that all inputs will have the same rotation)
            transformd_x0 = self.input_transform_net(x0)  # transformd_x0: [3, 3]
            x = torch.matmul(transformd_x0, x)

        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        x, feat_mask = get_graph_feature(x, self.args, k=self.k, mask=mask)  # x: [b, 6, 1024, 20]
        # process point and inflate it from 6 to e.g., 64
        if self.time_cond:
            x = self.conv1(x, t_emb)
        else:
            x = self.conv1(x)  # x: [b, 64, 1024, 20]
        if feat_mask is not None:
            x = torch.where(feat_mask.expand_as(x).bool(), x,
                    torch.ones_like(x).fill_(-np.inf))
        # per each feature (from e.g., 64) take the max value from the representative vectors
        # Conceptually this means taking the neighbor that gives the highest feature value.
        # returns a tensor of size e.g., (batch_size, 64, #points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x, feat_mask = get_graph_feature(x1, self.args, k=self.k, mask=mask)
        if self.time_cond:
            x = self.conv2(x, t_emb)
        else:
            x = self.conv2(x)
        if feat_mask is not None:
            x = torch.where(feat_mask.expand_as(x).bool(), x,
                    torch.ones_like(x).fill_(-np.inf))
            #x[~feat_mask.expand_as(x).bool()] = -np.inf
        x2 = x.max(dim=-1, keepdim=False)[0]

        x, feat_mask = get_graph_feature(x2, self.args, k=self.k, mask=mask)
        if self.time_cond:
            x = self.conv3(x, t_emb)
        else:
            x = self.conv3(x)
        if feat_mask is not None:
            x = torch.where(feat_mask.expand_as(x).bool(), x,
                    torch.ones_like(x).fill_(-np.inf))
            #x[~feat_mask.expand_as(x).bool()] = -np.inf
        x3 = x.max(dim=-1, keepdim=False)[0]

        x, feat_mask = get_graph_feature(x3, self.args, k=self.k, mask=mask)
        if self.time_cond:
            x = self.conv4(x, t_emb)
        else:
            x = self.conv4(x)
        if feat_mask is not None:
            x = torch.where(feat_mask.expand_as(x).bool(), x,
                    torch.ones_like(x).fill_(-np.inf))
            #x[~feat_mask.expand_as(x).bool()] = -np.inf
        x4 = x.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1) # [B, feat_dim, N]
        x5 = self.conv5(x_cat)  # [b, 1024, 1024]
        #cls_logits['features'] = x5
        x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
        cls_logits['features'] = x5
        if mask is not None:
            x5_ = x5.clone()
            x5_[~mask.unsqueeze(1).expand_as(x5)] = -np.inf
        else:
            x5_ = x5
        x1 = F.adaptive_max_pool1d(x5_, 1).view(batch_size, -1)
        if mask is not None:
            x5__ = x5.clone()
            x5__[~mask.unsqueeze(1).expand_as(x5)] = 0
        else:
            x5__ = x5
        x2 = F.adaptive_avg_pool1d(x5__, 1).view(batch_size, -1)
        if mask is not None:
            x2 = x2 * num_points / mask.float().sum(dim=1, keepdim=True)
        x = torch.cat((x1, x2), 1)

        # x5 = F.leaky_relu(self.bn5(x), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        # x5_pool = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        # x = x5_pool

        cls_logits["cls"] = self.cls_C(x)
        if alpha != 0:
            reverse_x = ReverseLayerF.apply(x, alpha)
        else:
            reverse_x = x
        if hasattr(self, 'cl_head'):
            cls_logits['cl_feat'] = self.cl_head(x)
        else:
            cls_logits['cl_feat'] = l2_norm(x, 1) # without head
        cls_logits["domain_cls"] = self.domain_C(reverse_x)
        cls_logits["rot_cls1"] = self.rotcls_C1(x)
        cls_logits["rot_cls2"] = self.rotcls_C2(x)
        cls_logits["def_cls"] = self.defcls_C(x)
        # cls_logits["curv_conf"] = self.curvconfreg_C(x)
        # cls_logits["norm_reg"] = self.normreg_C(x5).permute(0, 2, 1)
        cls_logits["decoder"] = self.DecoderFC(x)
        if hasattr(self, 'stat_C'):
            cls_logits['stat'] = self.stat_C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            cls_logits["DefRec"] = self.DefRec(DefRec_input)

        return cls_logits


class class_classifier(nn.Module):
    def __init__(self, args, input_dim, num_class=10):
        super(class_classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(input_dim, 512, bias=bias, activation=activate,
                bn=True, norm=getattr(args, 'fc_norm', True))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True
            , norm=getattr(args, 'fc_norm', True)
                )
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits


class ssl_classifier(nn.Module):
    def __init__(self, args, input_dim, num_class):
        super(ssl_classifier, self).__init__()
        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        self.mlp1 = fc_layer(input_dim, 256,
            activation=activate,
            norm=getattr(args, 'fc_norm', True)
                )
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        feats = self.mlp2(x)
        feats = l2_norm(feats, 1)
        return feats


class linear_classifier(nn.Module):
    def __init__(self, input_dim, num_class):
        super(linear_classifier, self).__init__()
        self.mlp1 = nn.Linear(input_dim, num_class)

    def forward(self, x):
        logits = self.mlp1(x)
        return logits


class domain_classifier(nn.Module):
    def __init__(self, args, input_dim, num_class=2):
        super(domain_classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(input_dim, 512, bias=bias, activation=activate,
                bn=True,
            norm=getattr(args, 'fc_norm', True)
                )
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True,
            norm=getattr(args, 'fc_norm', True)
                )
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.mlp1(x)
        x2 = self.mlp2(x)
        logits = self.mlp3(x2)
        return logits


class DecoderFC(nn.Module):
    def __init__(self, args, input_dim):
        super(DecoderFC, self).__init__()
        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(input_dim, 512, bias=bias, activation=activate,
                bn=True,
            norm=getattr(args, 'fc_norm', True)
                )
        self.mlp2 = fc_layer(512, 512, bias=True, activation=activate, bn=True,
            norm=getattr(args, 'fc_norm', True)
                )
        self.mlp3 = nn.Linear(512, args.output_pts * 3)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x


class RegionReconstruction(nn.Module):
    """
    Region Reconstruction Network - Reconstruction of a deformed region.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    """

    def __init__(self, args, input_size):
        super(RegionReconstruction, self).__init__()
        self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.conv4(x)
        return x.permute(0, 2, 1)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
