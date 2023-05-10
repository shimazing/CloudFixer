import pandas as pd
import copy
import utils
import random
import argparse
import wandb
from os.path import join
from build_model import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from train_test import train_epoch, test

from data.dataloader_Norm import ScanNet, ModelNet, ShapeNet, label_to_idx, \
    NUM_POINTS, ModelNet40C
from data.dataloader_Norm import ShapeNetCore
from data.dataloader_Norm import idx_to_label
from utils_GAST import pc_utils_Norm
from utils_GAST.pc_utils_Norm import rotate_shape_tensor #, log
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from Models_Norm import PointNet, DGCNN
from dgcnn_modelnet40 import DGCNN as DGCNN_modelnet40
from utils_GAST.pc_utils_Norm import scale_to_unit_cube_torch, farthest_point_sample
#from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F
from voxelization_guide import Voxelization
import log
from torch.cuda.amp import custom_bwd, custom_fwd


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label = None,
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

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if not isinstance(dataset, ModelNet40C):
            if dataset.partition == 'train':
                return dataset.label[dataset.train_ind]
            elif dataset.partition == 'val':
                return dataset.label[dataset.val_ind]
        return dataset.label.squeeze()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='pointnet',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics | '
                         'pointnet')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine, linear')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_nodes', type=int, default=1024)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=256, #128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str, default='mazing')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb') # TODO
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str,
        default="outputs/pointnet/generative_model_last.npy", #required=True,
                    #help='outputs/unit_val_shapenet_pointnet_resume/generative_model_last.npy'
                    )
parser.add_argument('--dynamics_config', type=str,
        default='pointnet2/exp_configs/mvp_configs/config_standard_ori.json')


parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
#parser.add_argument('--ema_decay', type=float, default=0.9999,
#                    help='Amount of EMA decay, 0 means off. A reasonable value'
#                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 1, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--jitter', type=eval, default=False)
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='mean',
                    help='"sum" or "mean"')
parser.add_argument('--out_path', type=str, default='./exps')
parser.add_argument('--knn', type=int, default=32)
parser.add_argument('--accum_grad', type=int, default=1)
parser.add_argument('--t', type=float, default=0.4)
parser.add_argument('--t_thrs', type=float, default=0.0)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--voting', type=str, default='hard', choices=['hard', 'soft'])
parser.add_argument('--accum_edit', action='store_true', help='Disable wandb') # TODO
parser.add_argument('--scale_mode', type=str, default='unit_std')
parser.add_argument('--cls_scale_mode', type=str, default='unit_std')
parser.add_argument('--scale', type=float, default=3)
########################
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--output_pts', type=int, default=512)

######################## guided sampling
parser.add_argument('--guidance_scale', type=float, default=0)
parser.add_argument('--mode', nargs='+', type=str, default=['eval'])
parser.add_argument('--dataset', type=str, default='shapenet')
parser.add_argument('--keep_sub', type=eval, default=False) # TODO
parser.add_argument('--no_zero_mean', action='store_true') # TODO
parser.add_argument('--n_subsample', type=int, default=128)
parser.add_argument('--classifier', type=str,
    default='../GAST/experiments/GAST_balanced_unitnorm_randomscale0.2/model.ptdgcnn')
parser.add_argument('--self_ensemble', action='store_true')
parser.add_argument('--egsde', action='store_true')
parser.add_argument('--domain_cls', type=str, default='outputs/domain_classifier_DGCNN_shape_model_timecondGN.pt')
parser.add_argument('--entropy_guided', action='store_true')
parser.add_argument('--lambda_s', default=100, type=float)
parser.add_argument('--lambda_ent', default=100, type=float)
parser.add_argument('--lambda_i', default=1, type=float)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--ddim', action='store_true')
parser.add_argument('--dpm_solver', action='store_true')
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--latent_subdist', action='store_true')
parser.add_argument('--n_inversion_steps', type=int, default=50) #action='store_true')
parser.add_argument('--n_reverse_steps', type=int, default=20) #action='store_true')
parser.add_argument('--voxel_resolution', type=int, default=32)
parser.add_argument('--voxelization', action='store_true')
parser.add_argument('--time_cond', action='store_true')
parser.add_argument('--gn', type=eval, default=False) # action='store_true')
parser.add_argument('--preprocess_model', type=str,
    default='outputs/stat_pred_only_lr1e-4_nregions5_rng3.5_droprate0.7/model.ptdgcnn')
parser.add_argument('--nregions', type=int, default=5)
parser.add_argument('--input_transform', action='store_true',
    help='whether to apply input_transform (rotation) in DGCNN')
parser.add_argument('--temperature', type=float, default=2.5)
parser.add_argument('--noise_t0', action='store_true')
parser.add_argument('--bn', default='bn', type=str)
parser.add_argument('--radius', default=0.2, type=float)
parser.add_argument('--ilvr', default=0, type=float)
# for sds loss test
parser.add_argument('--pre_trans', action='store_true')
parser.add_argument('--activate_mask', action='store_true')
parser.add_argument('--latent_trans', action='store_true')
parser.add_argument('--random_trans', default=0, type=float)
parser.add_argument('--n_update', default=1, type=int)
parser.add_argument('--lr', type=float, default=1e-2) #2e-4)
parser.add_argument('--beta1', type=float, default=0.5) #2e-4)
parser.add_argument('--beta2', type=float, default=0.999) #2e-4)
parser.add_argument('--l1', type=float, default=0) #2e-4)
parser.add_argument('--matching_t', type=float, default=0.2) #2e-4)
parser.add_argument('--weight_decay', type=float, default=0) #2e-4)
parser.add_argument('--n_iters_per_update', type=int, default=1) #2e-4)
parser.add_argument('--accum', type=int, default=1) #2e-4)
parser.add_argument('--optim', type=str, default='adamw') #2e-4)
parser.add_argument('--subsample', type=int, default=1024) #2e-4)

args = parser.parse_args()
if 'eval' in args.mode:
    args.no_wandb = True

ori_time_cond = args.time_cond
zero_mean = not args.no_zero_mean
args.cuda = not args.no_cuda and torch.cuda.is_available()
#dtype = torch.float32
device = torch.device("cuda" if args.cuda else "cpu")

io = log.IOStream(args)
voxelization = Voxelization(resolution=args.voxel_resolution)

def split_set(dataset, domain='scannet', set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3

    return out

def compute_rotation_matrix_from_ortho6d(poses): # 6DRepNet (https://github.com/thohemp/6DRepNet/blob/385008b04f32713347e689b27b92067cba710c42/sixdrepnet/utils.py#L144)
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix



def latent_sds_loss_ver3(x0_est, t, trans, log_scale, rot, w=None):
    assert t is not None
    #gamma_t = model.inflate_batch_array(model.gamma(t), zt)
    #alpha_t = model.alpha(gamma_t, zt)
    #sigma_t = model.sigma(gamma_t, zt)
    #with torch.no_grad():
    #    node_mask = zt.new_ones(zt.shape[:2]).to(zt.device).unsqueeze(-1)
    #    eps_t, x0_est = model_dp(zt+alpha_t*trans, t=t, phi=True, return_x0_est=True, node_mask=node_mask)
    loss = sds_loss((x0_est+trans)@rot, t=t) #None)
    return loss #, eps_t, x0_est

@torch.enable_grad()
def latent_trans_ver3(zt, t, x, lr=args.lr, steps=args.n_iters_per_update,
        verbose=False, accum=args.accum):
    gamma_t = model.inflate_batch_array(model.gamma(t), zt)
    alpha_t = model.alpha(gamma_t, zt)
    sigma_t = model.sigma(gamma_t, zt)
    with torch.no_grad():
        node_mask = zt.new_ones(zt.shape[:2]).to(zt.device).unsqueeze(-1)
        eps_t, x0_est = model_dp(zt, t=t, phi=True, return_x0_est=True, node_mask=node_mask)
    trans = torch.nn.Parameter(zt.new_zeros((zt.size(0),1,3)))
    rotation = zt.new_zeros((zt.size(0), 6))
    rotation[:, 0] = 1
    rotation[:, 4] = 1
    rotation = torch.nn.Parameter(rotation)
    log_scale = torch.nn.Parameter(zt.new_zeros((zt.size(0),1,1)))
    optim = torch.optim.SGD([{'params': [trans,
        rotation], 'lr':lr},
                        #{'params':[log_scale], 'lr': 0.1*lr},
                            ],
                            lr=lr, weight_decay=0)
    for step in tqdm(range(steps), desc=f'latent trans ver.2 lr={lr}'):
        rot = compute_rotation_matrix_from_ortho6d(rotation)
        #loss, eps_t, x0_est = latent_sds_loss_ver3(zt, t, trans, log_scale)
        loss = latent_sds_loss_ver3(x0_est, t, trans, log_scale, rot)
        #yt = zt + alpha_t * trans
        #x_trans = x + trans
        #loss = latent_sds_loss_ver2(yt, t, x_trans)
        (loss/(1 if not accum else accum)).backward()
        #print(trans.grad.squeeze(), trans.shape)
        if not accum or ((step +1) % accum == 0):
            optim.step()
            optim.zero_grad()
    if steps % accum: # 나누어떨어지지 않으면
        optim.step()
        optim.zero_grad()
    rot = compute_rotation_matrix_from_ortho6d(rotation)
    new_x = (x+trans)@rot
    new_zt = (alpha_t*(x0_est + trans) + sigma_t*eps_t)@rot
    #new_zt = log_scale.exp() * alpha_t * x0_est + alpha_t * trans + sigma_t * eps_t
    return new_zt.detach(), new_x # trans.detach(), log_scale.exp().detach()


def latent_sds_loss_ver2(zt, t, x, w=None):
    assert t is not None
    gamma_t = model.inflate_batch_array(model.gamma(t), zt)
    alpha_t = model.alpha(gamma_t, zt)
    sigma_t = model.sigma(gamma_t, zt)

    with torch.no_grad():
        eps = (zt - alpha_t*x)/sigma_t
        node_mask = zt.new_ones(zt.shape[:2]).to(zt.device).unsqueeze(-1)
        pred_noise = model_dp(zt, t=t, node_mask=node_mask, phi=True)
    if w is None:
        w = sigma_t.pow(2) #alpha_t / sigma_t #sigma_t.pow(2)
    grad = w*(pred_noise - eps)
    grad = torch.nan_to_num(grad)
    loss = SpecifyGradient.apply(x, grad)
    #eps_t, x0_est = model_dp(zt, t=t, phi=True, return_x0_est=True, node_mask=node_mask)
    #loss = sds_loss(x0_est, t=t, w=None) #alpha_t/sigma_t)
    return loss

@torch.enable_grad()
def latent_trans_ver2(zt, t, x, lr=1e-3, steps=1, verbose=False):
    print("trans")
    zt = zt.detach()
    x = x.detach()
    gamma_t = model.inflate_batch_array(model.gamma(t), zt)
    alpha_t = model.alpha(gamma_t, zt)
    sigma_t = model.sigma(gamma_t, zt)
    trans = torch.nn.Parameter(zt.new_zeros((zt.size(0),1,3)))
    scale = torch.nn.Parameter(zt.new_ones((zt.size(0),1,1)))
    optim = torch.optim.SGD([{'params': [trans], 'lr':lr},
                        {'params':[scale], 'lr': lr},
                            ],
                            lr=lr, weight_decay=0)
    eps_true = (zt - alpha_t*x) / sigma_t
    for step in range(steps): #tqdm(range(steps), desc=f'latent trans ver.2 lr={lr}'):
        x_trans = scale.clamp(min=0)*x + trans
        yt = alpha_t*(x_trans) + sigma_t*eps_true
        #yt = scale.clamp(min=0)*zt + alpha_t * trans
        #x_trans = scale.clamp(min=0)*x - (1-scale.clamp(min=0))*sigma_t/alpha_t*eps_true + trans
        #x_trans = x + trans
        loss = latent_sds_loss_ver2(yt, t, x_trans)
        loss.backward()
        optim.step()
        optim.zero_grad()
    new_x = scale.clamp(min=0)*x + trans
    new_zt = alpha_t*(new_x) + sigma_t*eps_true
    #new_zt = zt+alpha_t*trans.detach()
    #new_x = scale.clamp(min=0)*x - (1-scale.clamp(min=0))*sigma_t/alpha_t*eps_true + trans
    return new_zt, new_x
    #return (zt + alpha_t * trans).detach(), trans.detach()


def latent_sds_loss(zt, t):
    assert t is not None
    gamma_t = model.inflate_batch_array(model.gamma(t), zt)
    alpha_t = model.alpha(gamma_t, zt)
    sigma_t = model.sigma(gamma_t, zt)
    node_mask = zt.new_ones(zt.shape[:2]).to(zt.device).unsqueeze(-1)
    eps_t, x0_est = model_dp(zt, t=t, phi=True, return_x0_est=True, node_mask=node_mask)
    loss = sds_loss(x0_est, t=t, w=None) #alpha_t/sigma_t)
    return loss

@torch.enable_grad()
def latent_trans(zt, t, lr=5e-2, steps=5, verbose=False):
    gamma_t = model.inflate_batch_array(model.gamma(t), zt)
    alpha_t = model.alpha(gamma_t, zt)
    trans = torch.nn.Parameter(zt.new_zeros((zt.size(0),1,3)))
    log_scale = torch.nn.Parameter(zt.new_zeros((zt.size(0),1,1)))
    optim = torch.optim.Adam([{'params': [trans], 'lr':lr},
                        #{'params':[log_scale], 'lr': 0.0*lr},
                            ],
                            lr=lr)
    for step in tqdm(range(steps), desc=f'pre trans lr={lr}', ncols=100):
        yt = zt + trans
        loss = latent_sds_loss(yt, t)
        loss.backward()
        optim.step()
        optim.zero_grad()
    return (zt + trans).detach(), (trans/alpha_t).detach()


def sds_loss(x, t=None, w=None):
    if t is None:
        t = 0.02 + 0.2 * torch.rand(x.shape[0], 1).to(x.device)
    gamma_t = model.inflate_batch_array(model.gamma(t), x)
    alpha_t = model.alpha(gamma_t, x)
    sigma_t = model.sigma(gamma_t, x)

    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
        n_nodes=x.size(1),
        node_mask=node_mask,
    )
    with torch.no_grad():
        z_t = x*alpha_t + eps*sigma_t
        pred_noise = model_dp(z_t, t=t, node_mask=node_mask, phi=True)
    if w is None:
        w = sigma_t.pow(2) #alpha_t / sigma_t #sigma_t.pow(2)
    grad = w*(pred_noise - eps)
    grad = torch.nan_to_num(grad)
    loss = SpecifyGradient.apply(x, grad)
    #loss.requires_grad_(True)
    return loss

def matching_loss(x, t=None, w=None):
    if t is None:
        t = 0.02 + args.matching_t * torch.rand(x.shape[0], 1).to(x.device)
    gamma_t = model.inflate_batch_array(model.gamma(t), x)
    alpha_t = model.alpha(gamma_t, x)
    sigma_t = model.sigma(gamma_t, x)

    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
        n_nodes=x.size(1),
        node_mask=node_mask,
    )
    z_t = x*alpha_t + eps*sigma_t
    pred_noise = model_dp(z_t, t=t, node_mask=node_mask, phi=True)
    loss = (pred_noise - eps).pow(2).mean()
    return loss

@torch.enable_grad()
def pre_trans(x, mask, lr=1e-1, steps=10000, verbose=True):
    trans = torch.nn.Parameter(x.new_zeros((x.size(0),1,3)))
    scale = torch.nn.Parameter(x.new_ones((x.size(0),1,1)))
    delta = torch.nn.Parameter(torch.zeros_like(x))

    rotation = x.new_zeros((x.size(0), 6))
    rotation[:, 0] = 1
    rotation[:, 4] = 1
    rotation = torch.nn.Parameter(rotation)

    optim = torch.optim.SGD([
        {'params': [trans,
            #rotation
        ], 'lr':0*lr, 'weight_decay': 0},
        {'params': delta, 'lr':lr, 'weight_decay':0},
        {'params': shear, 'lr':0*lr, 'weight_decay':0},
                        #{'params':[scale], 'lr': lr},
                            ],
                            lr=lr)
    for step in tqdm(range(steps), desc='pre trans', ncols=100):
        rot = compute_rotation_matrix_from_ortho6d(rotation)
        y = (x + trans+ delta*(1-mask)) @ shear_inv(shear) @ rot
        #y = (x + trans+ delta) @ rot
        loss = sds_loss(y)
        (loss+0.0*torch.norm(delta, p=1)).backward()
        #torch.nn.utils.clip_grad_norm_([trans, rot], 1)
        optim.step()
        optim.zero_grad()
        if verbose and step % 10 == 0:
            print()
            print(step, "delta.abs()", delta.abs().mean())
            print(step, "trans.abs()", trans.abs().mean())
            print(step, "mean", y.mean(dim=1).abs().mean())
            print(step, "scale", y.flatten(1).std(1).mean())
    rot = compute_rotation_matrix_from_ortho6d(rotation)
    return ((x+trans+delta*(1-mask)) @ shear_inv(shear) @ rot).detach()


def shear_inv(shear):
    b = shear[:, 0]
    d = shear[:, 1]
    e = shear[:, 2]
    f = shear[:, 3]
    ones = torch.ones_like(b)
    zeros = torch.zeros_like(b)
    return torch.stack([
        torch.stack((ones, zeros, -b), dim=-1),
        torch.stack((-d + e*f, ones, b*d - e), dim=-1),
        torch.stack((-f, zeros, ones), dim=-1)
        ], dim=1) / (1 - (b*f)[:, None, None]) # batch x 3 x 3


#def pre_trans_ver2(x, mask, lr=5e-2, steps=200, verbose=True):
@torch.enable_grad()
def pre_trans_ver2(x, mask, lr=args.lr, steps=args.n_update, verbose=True, activate_mask=args.activate_mask):
    if args.model == 'pvd':
        model.train()
    if not activate_mask:
        mask[:] = 0
    else:
        print(mask.sum(dim=1).flatten())
    trans = torch.nn.Parameter(x.new_zeros((x.size(0),1,3)))
    scale = torch.nn.Parameter(x.new_zeros((x.size(0),1,1)))
    delta = torch.nn.Parameter(torch.zeros_like(x))
    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)

    rotation = x.new_zeros((x.size(0), 6))
    rotation[:, 0] = 1
    rotation[:, 4] = 1
    rotation = torch.nn.Parameter(rotation)

    shear = x.new_zeros((x.size(0), 4)) # b, d, e, f
    shear = torch.nn.Parameter(shear)

    if args.optim == 'adamw':
        optim = torch.optim.AdamW([
            {'params': [delta], 'lr':(0 if (('shear' in args.dataset) or ('original' in
                args.dataset)) else 1) * lr}, #, 'weight_decay':0.0},
            {'params':[rotation], 'lr': (1 if 'rotation' in args.dataset or
            'distortion' in args.dataset else 0)
                *lr, 'weight_decay': 0.0},
            {'params': [shear], 'lr':(1 if 'shear' in args.dataset else 0) * lr},
                                ],
                                lr=lr, weight_decay=args.weight_decay,
                                betas=(args.beta1, args.beta2))
    else:
        optim = torch.optim.SGD([
            #{'params': [delta], 'lr':(0 if (('shear' in args.dataset) or ('original' in
            #    args.dataset)) else 1) * lr}, #, 'weight_decay':0.0},
            {'params':[rotation], 'lr': 100  #(1 if 'rotation' in args.dataset else 0)
                *lr, 'weight_decay': 0.0},
            #{'params': [shear], 'lr':(1 if 'shear' in args.dataset else 0) * lr, 'weight_decay':0},
                                ],
                                lr=lr, weight_decay=args.weight_decay
                                )
    for step in tqdm(range(steps), desc='pre trans', ncols=100):
        rot = compute_rotation_matrix_from_ortho6d(rotation)
        #y = 2*torch.sigmoid(scale) * x
        #y = scale.exp() * x + trans + delta * (1-mask)
        y = x + delta * (1-mask)
        if 'original' not in args.dataset or 'rotation' not in args.dataset:
            y = y - y.mean(dim=1, keepdim=True)
            y = y / y.flatten(1).std(dim=1, keepdim=True)[:, :, None]
        y = y @ shear_inv(shear)
        y = y @ rot
        loss = matching_loss(y) + args.l1 *torch.norm(delta, 1) / args.batch_size / y.shape[1]
        #loss = sds_loss(y, w=1)
        #t = x.new_zeros((x.shape[0], 1), device=x.device)
        #with torch.no_grad():
        #    eps_t, x0_est = model_dp(y, t=t, phi=True, return_x0_est=True, node_mask=node_mask)
        #loss = SpecifyGradient.apply(y, eps_t)
        #loss = eps_t.flatten(1).pow(2).mean()
        loss.backward()
        #assert not torch.all(delta.grad == 0)
        optim.step()
        optim.zero_grad()
        if verbose and step % 10 == 0:
            print()
            print(step, "delta.abs()", delta.abs().mean())
            print(step, "rotation", rotation.abs().mean(dim=0))
            #print(scale.exp().flatten())
            #print(scale.exp().flatten())
            #print((1+scale).clamp(min=0).flatten())
            #print(step, "trans.abs()", trans.abs().mean())
            #print(step, "mean", y.mean(dim=1).abs().mean())
            #print(step, "scale", y.flatten(1).std(1).mean())
    rot = compute_rotation_matrix_from_ortho6d(rotation)
    #return ((scale.clamp(min=0)*x+trans+delta*(1-mask)) @ rot).detach()
    #return scale.exp()
    #return 2*torch.sigmoid(scale)
    if args.model == 'pvd':
        model.eval()
    y = (x + delta * (1-mask))
    if 'original' not in args.dataset or 'rotation' not in args.dataset:
        y = y - y.mean(dim=1, keepdim=True)
        y = y / y.flatten(1).std(dim=1, keepdim=True)[:, :, None]
    y = y @ shear_inv(shear) @ rot
    return y
    #2*torch.sigmoid(scale) #(x*scale.clamp(min=0)).detach()


if args.n_nodes == 1024:
    dataset_ = ShapeNet(io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=True, zero_mean=zero_mean)
    if args.dataset == 'scannet':
        test_dataset = ScanNet(io, './data', 'test' if not 'aug' in args.mode
                else 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=False, zero_mean=zero_mean) # for classification
        # for preprocessing
        if args.preprocess:
            ori_model = args.model
            args.model = 'dgcnn'
            args.pred_stat = True
            args.time_cond = False
            args.fc_norm = True
            preprocess_model = DGCNN(args).to(device)
            preprocess_model.load_state_dict(torch.load(args.preprocess_model,
                map_location='cpu'))
            preprocess_model.eval()
            args.model = ori_model
            args.time_cond = ori_time_cond
            args.pred_stat = False
    elif args.dataset == 'modelnet':
        test_dataset = ModelNet(io, './data', 'test' if not 'aug' in args.mode
                else 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=False, zero_mean=zero_mean) # for classification
    elif args.dataset == 'shapenet':
        test_dataset = ShapeNet(io, './data', 'test' if not 'aug' in args.mode
                else 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=False, zero_mean=zero_mean) # for classification
    elif args.dataset.startswith('modelnet40c'):
        test_dataset = ModelNet40C(split='test',
                num_classes=40,
                corruption='_'.join(args.dataset.split('_')[1:-1]),
                severity=args.dataset.split('_')[-1].split('.')[0],
                random_trans=args.random_trans,
                rotate=True, #False if not args.time_cond else True,
                random_rotation=False,
                subsample=args.subsample,
                )
        args.num_classes = 40
    # TODO jitter??!!
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader_vis = DataLoader(test_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False,
            sampler=ImbalancedDatasetSampler(test_dataset))
else: # 2048
    train_dset = ShapeNetCore(
        path='data/shapenet.hdf5', #args.dataset_path,
        cates=['airplane'], #args.categories,
        split='train',
        scale_mode='shape_unit', #args.scale_mode,
    )
    val_dset = ShapeNetCore(
        path='data/shapenet.hdf5', #args.dataset_path,
        cates=['airplane'], #args.categories,
        split='val',
        scale_mode='shape_unit', #args.scale_mode,
    )

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=2*args.batch_size, shuffle=False)



args.wandb_usr = utils.get_wandb_username(args.wandb_usr)


if args.resume is not None:
    exp_name = args.exp_name #+ '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    #with open(join(args.resume, 'args.pickle'), 'rb') as f:
    #    args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method
    io.cprint(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name + '_vis', 'project':
        'sdedit', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Create EGNN flow
model = get_model(args, device)
if args.resume is not None:
    model.load_state_dict(torch.load(args.resume))

if args.dpm_solver:
    import dpm_solver
    alphas_cumprod = torch.tensor(model.gamma.alphas2).to(device)
    ns = dpm_solver.NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

if args.dataset.startswith('modelnet40') and not args.time_cond:
    classifier = DGCNN_modelnet40()
    classifier.load_state_dict(torch.load('outputs/dgcnn_modelnet40_best_test.pth')['model_state'])
    classifier = torch.nn.DataParallel(classifier)
    classifier.to(device).eval()
    print("load dgcnn_modelnet40")
else:
    ori_args_model = args.model
    args.model = 'dgcnn'
    #args.time_cond = False
    if True: #args.time_cond:
        args.fc_norm = False
        args.cls_scale_mode = 'unit_std'

    args.nregions = 3  # (default value)
    #ori_gn = args.gn
    #args.gn = True

    classifier = DGCNN(args).to(device).eval()
    state_dict = torch.load(
            args.classifier, map_location='cpu')
    classifier = torch.nn.DataParallel(classifier)
    if args.bn == 'bn': # and not args.dataset.startswith('modelnet40c'):
        print("bn!! sync mode")
        from sync_batchnorm import convert_model
        classifier = convert_model(classifier).to(device)
    classifier.load_state_dict(
        torch.load(
            args.classifier, map_location='cpu'),
            strict=True
    )
    classifier.eval()
    args.model = ori_args_model
    #args.gn = ori_gn

if args.lambda_s > 0:
    from chamfer_distance import ChamferDistance as chamfer_dist
    chamfer_dist_fn = chamfer_dist()
    ori_model = args.model
    args.model = 'dgcnn'
    ori_time_cond = args.time_cond
    args.time_cond = True
    args.fc_norm = False
    ori_input_transform = args.input_transform
    args.input_transform = False
    ori_bn = args.bn
    args.bn = 'ln'
    domain_cls = DGCNN(args).eval().to(device)
    args.input_transform = ori_input_transform
    if args.domain_cls:
        domain_cls_state_dict = torch.load(
            args.domain_cls,
            map_location='cpu')
        #keys = list(domain_cls_state_dict.keys())
        #for key in keys:
        #    if 'input_transform_net' in key:
        #        domain_cls_state_dict.pop(key)
        domain_cls.load_state_dict(
            domain_cls_state_dict, strict=False
            )
    args.bn = ori_bn
    args.time_cond = ori_time_cond
    args.model = ori_model
    domain_cls = torch.nn.DataParallel(domain_cls)
    #model.domain_cls = domain_cls
model = model.to(device)

#if args.entropy_guided:
from chamfer_distance import ChamferDistance as chamfer_dist
chamfer_dist_fn = chamfer_dist()


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def softmax(x):
    max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def get_color(coords, corners=np.array([
                                    [-1, -1, -1],
                                    [-1, 1, -1],
                                    [-1, -1, 1],
                                    [1, -1, -1],
                                    [1, 1, -1],
                                    [-1, 1, 1],
                                    [1, -1, 1],
                                    [1, 1, 1]
                                ]) * args.scale,
    ):
    coords = np.array(coords) # batch x n_points x 3
    corners = np.array(corners) # n_corners x 3
    colors = np.array([
        [255, 0, 0],
        [255, 127, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [75, 0, 130],
        [143, 0, 255],
    ])

    dist = np.linalg.norm(coords[:, :, None, :] -
            corners[None, None, :, :], axis=-1)

    weight = softmax(-dist)[:, :, :, None] #batch x NUM_POINTS x n_corners x 1
    rgb = (weight * colors).sum(2).astype(int) # NUM_POINTS x 3
    return rgb

def preprocess(data):
    print("preprocess")
    pc = data[4].to(device)
    batch_size = len(pc)
    num_points = data[3]
    mask = torch.zeros((len(data[0]), 1024, 1)).to(device)
    mask[num_points.view(-1, 1) > torch.arange(1024)] = 1
    logits = preprocess_model(pc.permute(0,2,1).float(), activate_DefRec=False,
            mask=mask.view(batch_size, -1).bool())
    pred_stat = logits['stat']
    pc = data[0].to(device)
    data_moved = (pc*pred_stat[:, 3:].unsqueeze(1).exp() + pred_stat[:,
        :3].unsqueeze(1)) #* mask
    new_pc = data_moved
    mask = new_pc.new_ones(new_pc.shape[:-1]).bool() # obsolete
    #new_noise_pc = np.random.normal(-pred_stat[:, :3].view(-1, 1, 3).cpu().numpy(), scale=pred_stat[:,
    #    3:].exp().view(-1, 1, 1).cpu().numpy(), size=(batch_size, 1024, 3))
    #new_noise_pc = torch.tensor(new_noise_pc).to(device)
    #n_new = torch.maximum(1024 - num_points, torch.ones_like(num_points)*200)
    #new_noise_pc[torch.arange(1024) > n_new.view(-1, 1)] = 0
    #new_pc = torch.cat((pc, new_noise_pc), dim=1).float()
    #chosen = farthest_point_sample(new_pc.permute(0,2,1), npoint=1024)[0]
    #new_pc = new_pc[torch.arange(batch_size).view(-1, 1), chosen]
    return pc, new_pc, mask #chosen < 1024


model.eval()
model_dp = torch.nn.DataParallel(model)
def main():
    K = args.K

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    guide = args.lambda_i > 0 or args.lambda_s > 0 or args.lambda_ent > 0 or \
        args.keep_sub or args.ilvr > 0

    if 'eval' in args.mode or 'aug' in args.mode:  # calc acc
      with torch.no_grad():
        if 'aug' in args.mode:
            pc_list = []
            label_list = []
        count = 0
        correct_count = [0 for _ in range(K+1)]
        correct_count_y = [0 for _ in range(K+1)]
        correct_count_self_ensemble = [0 for _ in range(K)]
        if args.ddim and ori_time_cond:
            correct_count_itmd = [[0, 0] for _ in range(K)]
        correct_count_vote = 0
        correct_count_vote_soft = 0
        correct_count_vote_soft_ensemble = 0
        correct_count_vote_y = 0
        correct_count_vote_soft_y = 0
        correct_count_vote_soft_ensemble_y = 0
        pred_change = np.zeros((10, 10, 10, args.K))
        x_edited_batch_list = []
        label_batch_list = []
        for iter_idx, data in tqdm(enumerate(test_loader)):
            if args.dataset.startswith('modelnet40') and not args.time_cond:
                x_ori = scale_to_unit_cube_torch(data[2].to(device))
                mask = data[3].to(device)
                print("get ori")
            else:
                x_ori = data[0].to(device)
            labels = data[1].to(device).flatten()
            if args.dataset == 'scannet' and args.preprocess:
                old_x, x, is_ori = preprocess(data)
                furthest_point_idx = \
                        farthest_point_sample((x*is_ori[:, :,
                            None].float()).permute(0,2,1), getattr(args,
                                'n_subsample', 64))[0]
                        #pointnet2_utils.furthest_point_sample(x * is_ori[:, :,
                        #    None].float(),
                        #    getattr(args, 'n_subsample', 64))
            else:
                # is_ori : batch_size x 1024
                x = data[0].to(device)
                if args.pre_trans:
                    #x = pre_trans(x, mask)
                    x = pre_trans_ver2(x, mask)
                if args.random_trans:
                    random_trans = data[2]
                furthest_point_idx = \
                    farthest_point_sample(x.permute(0,2,1), getattr(args,
                            'n_subsample', 64))[0]
                #furthest_point_idx = \
                #    pointnet2_utils.furthest_point_sample(x,
                #            getattr(args, 'n_subsample', 64))
            furthest_point_idx = furthest_point_idx.long()
            if args.ilvr:
                x_filtered = utils.mean_filter(x, K=50, radius=args.radius)
            #sub_x = \
            #    x[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]
            count += len(x)
            t = args.t * x.new_ones((x.shape[0], 1), device=x.device)
            node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
            gamma_t = model.inflate_batch_array(model.gamma(t), x)
            alpha_t = model.alpha(gamma_t, x)
            sigma_t = model.sigma(gamma_t, x)

            t0 = torch.zeros_like(t)
            gamma_t0 = model.inflate_batch_array(model.gamma(t0), x)
            alpha_t0 = model.alpha(gamma_t0, x)
            sigma_t0 = model.sigma(gamma_t0, x)
            noise_t0 = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )

            @torch.enable_grad()
            def cond_fn(yt, t, phi, x, furthest_point_idx, model_kwargs):
                if args.lambda_ent == 0 and args.lambda_s == 0 and args.lambda_i == 0:
                    return torch.zeros_like(yt)
                # Ss (realistic expert)
                classifier.requires_grad_(True)
                yt.requires_grad_(True)
                #if args.model == 'pvd':
                #    model.train()
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x).to(x)
                sigma_t = model.sigma(gamma_t, x).to(x)
                t_int = (t * model.T).long().float().flatten()

                entropy = 0
                domain_loss = 0
                if args.lambda_ent>0:
                    if not ori_time_cond:
                        eps = phi(yt, t, **model_kwargs)
                        y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                        #if args.cls_scale_mode == 'unit_norm':
                        #    y0_est_norm = scale_to_unit_cube_torch(y0_est)
                        cls_output = classifier(y0_est_norm) #.permute(0,2,1))
                        #else:
                        #    cls_output = classifier(y0_est.permute(0,2,1),
                        #            ts=torch.zeros_like(t).flatten())
                    else:
                        cls_output = classifier(yt.permute(0,2,1),
                                ts=t.flatten()*args.diffusion_steps)
                    cls_logits = cls_output['cls'] / args.temperature
                    cls_logits = cls_logits.to(x)
                    entropy = - (F.log_softmax(cls_logits, dim=-1) * F.softmax(cls_logits,
                        dim=-1)).sum(dim=-1).mean()
                if args.lambda_s > 0:
                    y_output = domain_cls(yt.permute(0,2,1), ts=t_int)
                    domain_y = y_output['domain_cls'].to(x)
                    feat_y = y_output['features'].permute(0,2,1)
                    domain_loss = F.cross_entropy(domain_y,
                            domain_y.new_zeros((len(domain_y),)).long())
                # Si (faithful expert)
                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=model_kwargs['node_mask'],
                    )
                if args.lambda_i > 0:
                    if args.voxelization:
                        if not args.latent_subdist: # in the sample space
                            eps = phi(yt, t, **model_kwargs)
                            x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                        furthest_point_idx]
                            y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                            vox_x, _, norm = voxelization(x.permute(0,2,1),
                                    x.permute(0,2,1))
                            vox_y = voxelization(y0_est.permute(0,2,1),
                                    y0_est.permute(0,2,1), norm=norm)[0]
                        else: # in the latent space
                            vox_x, _, norm = voxelization(xt.permute(0,2,1),
                                    xt.permute(0,2,1))
                            vox_y = voxelization(yt.permute(0,2,1),
                                    yt.permute(0,2,1), norm=norm)[0]
                        subdist_loss = F.mse_loss(vox_y, vox_x)
                    elif args.latent_subdist: # loss_i in latent space
                        xt = x * alpha_t + noise * sigma_t
                        xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        #subdist_loss = (xt_sub - yt_sub).pow(2).sum(-1).mean()
                        dist1, dist2, *_ = chamfer_dist_fn(xt_sub, yt)
                        #print(dist1.shape)
                        subdist_loss = dist1.mean()
                    else: # after model run
                        eps = phi(yt, t, **model_kwargs)
                        x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                        y0_est_sub = y0_est[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        #subdist_loss = (x_sub - y0_est_sub).pow(2).sum(-1).mean()
                        dist1_, dist2_, *_ = chamfer_dist_fn(x_sub, y0_est)
                        subdist_loss = dist1_.mean()
                else:
                    subdist_loss = 0
                grad = torch.autograd.grad(
                    (args.lambda_ent * entropy + args.lambda_s * domain_loss + args.lambda_i * subdist_loss),
                    yt, allow_unused=True)[0]
                #if args.model == 'pvd':
                #    model.eval()
                return grad


            #x_ori = x.detach().clone()

            #x_edit_list = [alpha_t0 * x + noise_t0 * sigma_t0 if args.noise_t0 else x]
            if False: #args.dataset.startswith('modelnet40') and not args.time_cond:
                x_edit_list_y = [x_ori]
            else:
                x_edit_list_y = [alpha_t0 * x + noise_t0 * sigma_t0 if args.noise_t0 else x]
            labels_list = [labels]
            if args.ddim and ori_time_cond:
                itmd_list = []
            for k in range(K):
                print(k+1,  "/", K)
                eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                if False: #args.ddim:
                    inversion_steps = np.linspace(0, args.t, args.n_inversion_steps)
                    # inversion
                    z = x.detach().clone()
                    print("Inversion")
                    for t, s in tqdm(zip(inversion_steps[:-1],
                        inversion_steps[1:]), total=len(inversion_steps)-1):
                        t_tensor = t * x.new_ones((x.shape[0], 1), device=x.device)
                        s_tensor = s * x.new_ones((x.shape[0], 1), device=x.device)
                        z = model_dp(z, sample_p_zs_given_zt_ddim=True, s=s_tensor,
                                t=t_tensor, node_mask=node_mask, edge_mask=None, cond_fn=None)
                else:
                    z = alpha_t * x + sigma_t * eps # diffusion

                y = z.detach().clone()
                if args.dpm_solver:
                    model_fn = dpm_solver.model_wrapper(
                        model_dp,
                        ns,
                        model_kwargs={'cond_fn': cond_fn if guide else None, 'phi':True,
                            'node_mask': node_mask, 'x_ori': x,
                            'furthest_point_idx': furthest_point_idx,
                            },
                        guidance_type='uncond',
                        )
                    solver = dpm_solver.DPM_Solver(model_fn,
                            noise_schedule=ns,
                            algorithm_type='dpmsolver',
                            )
                    z = solver.sample(z, steps=args.n_reverse_steps, t_start=args.t,
                            order=3,
                            skip_type='time_uniform',
                            method='singlestep_fixed',
                            lower_order_final=True,
                            denoise_to_zero=False,
                            solver_type='dpmsolver',
                    )
                    # TODO keep_sub for dpm_solver

                    if guide:
                        model_fn = dpm_solver.model_wrapper(
                            model_dp,
                            ns,
                            model_kwargs={'cond_fn': None, 'phi':True,
                                'node_mask': node_mask, 'x_ori': x,
                                },
                            guidance_type='uncond',
                            )
                        solver = dpm_solver.DPM_Solver(model_fn,
                                noise_schedule=ns,
                                algorithm_type='dpmsolver',
                                )
                        y = solver.sample(y, steps=args.n_reverse_steps, t_start=args.t,
                                order=3,
                                skip_type='time_uniform',
                                method='singlestep_fixed',
                                lower_order_final=True,
                                denoise_to_zero=False,
                                solver_type='dpmsolver',
                        )

                elif args.ddim:
                    reverse_steps = np.linspace(args.t*args.diffusion_steps, 0,
                            args.n_reverse_steps) / args.diffusion_steps
                    print("Reverse Steps")
                    if ori_time_cond:
                        itmd = [(z, args.t*args.diffusion_steps)] # list of (latent var, t)
                    if True: # verbose
                        print('z.mean', z.mean(dim=1).abs().mean())
                        print('x.mean', x.mean(dim=1).abs().mean())
                        print('z.std', z.flatten(1).std(1).mean())
                        print('x.std', x.flatten(1).std(1).mean())
                    for t, s in tqdm(zip(reverse_steps[:-1], reverse_steps[1:]),
                            total=len(reverse_steps)-1):
                        t_tensor = t * x.new_ones((x.shape[0], 1), device=x.device)
                        s_tensor = s * x.new_ones((x.shape[0], 1), device=x.device)
                        if args.latent_trans:
                            ver2 = False
                            if ver2:
                                z, x = latent_trans_ver2(z, t_tensor, x)
                            else: #ver3
                                for _ in range(args.n_update):
                                    z, x = latent_trans_ver3(z, t_tensor, x)
                            if True: # verbose
                                print('z.mean', z.mean(dim=1).abs().mean())
                                print('x.mean', x.mean(dim=1).abs().mean())
                                print('z.std', z.flatten(1).std())
                                print('x.std', x.flatten(1).std())
                        z = model_dp(z, sample_p_zs_given_zt_ddim=True, s=s_tensor,
                                t=t_tensor, node_mask=node_mask, edge_mask=None, #None,
                                cond_fn=cond_fn if guide and t>args.t_thrs else None,
                                #cond_fn=cond_fn_entropy if args.entropy_guided else (cond_fn_egsde if args.egsde else (cond_fn_sub
                                #if args.guidance_scale > 0 else None)), #noise=noise,
                                x_ori=x,
                                furthest_point_idx=furthest_point_idx,
                                )
                        if args.keep_sub and t > args.t_thrs:
                            # for sub_x
                            gamma_s = \
                                model.inflate_batch_array(model.gamma(s_tensor), x)
                            alpha_s = model.alpha(gamma_s, x)
                            sigma_s = model.sigma(gamma_s, x)
                            eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                                n_nodes=x.size(1),
                                node_mask=node_mask,
                                )
                            z[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                              furthest_point_idx] = (alpha_s * x + sigma_s * eps)[
                                    torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        if args.ilvr and t > args.t_thrs:
                            gamma_s = \
                                model.inflate_batch_array(model.gamma(s_tensor), x)
                            alpha_s = model.alpha(gamma_s, x)
                            sigma_s = model.sigma(gamma_s, x)
                            eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                                n_nodes=x.size(1),
                                node_mask=node_mask,
                                )
                            z_filtered = utils.mean_filter(z, K=50, radius=args.radius)
                            xs_filtered = x_filtered * alpha_s + sigma_s # utils.mean_filter(x * alpha_s + sigma_s * eps, K=50, radius=args.radius)
                            z = z + args.ilvr * (- z_filtered + xs_filtered)

                        if args.lambda_s > 0:
                            with torch.no_grad():
                                s_int = (s_tensor * args.diffusion_steps).long()
                                domain_logit = domain_cls(z.permute(0,2,1),
                                        ts=s_int.flatten())['domain_cls']
                                #print(domain_logit.argmax(dim=1))
                        if ori_time_cond:
                            itmd.append((z, s*args.diffusion_steps))
                        if guide: # args.egsde or args.entropy_guided:
                            y = model_dp(y, sample_p_zs_given_zt_ddim=True, s=s_tensor,
                                    t=t_tensor, node_mask=node_mask, edge_mask=None, #None,
                                    )
                    if ori_time_cond:
                        itmd_list.append(itmd)
                else:
                    t = args.t * x.new_ones((x.shape[0], 1), device=x.device)
                    for t_ in tqdm(range(int(args.t *
                        args.diffusion_steps))[::-1]):
                        if args.latent_trans:
                            z, trans_x = latent_trans(z, t)
                            x = x + trans_x
                        z, noise = model_dp(z, sample_p_zs_given_zt=True, s=t -
                                1./args.diffusion_steps, t=t,
                            node_mask=node_mask, edge_mask=None, #fix_noise=False,
                            cond_fn=cond_fn if guide else None,
                            #_entropy if args.entropy_guided else (cond_fn_egsde if args.egsde else (cond_fn_sub
                            #if args.guidance_scale > 0 else None)),
                            x_ori=x,
                            furthest_point_idx=furthest_point_idx,
                            return_noise=True)
                        # constraint (like ilvr)
                        if args.keep_sub and t_ > args.t_thrs * args.diffusion_steps:
                            # for sub_x
                            gamma_s = \
                                model.inflate_batch_array(model.gamma(t-1./args.diffusion_steps), x)
                            alpha_s = model.alpha(gamma_s, x)
                            sigma_s = model.sigma(gamma_s, x)
                            eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                                n_nodes=x.size(1),
                                node_mask=node_mask,
                                )
                            z[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                              furthest_point_idx] = (alpha_s * x + sigma_s * eps)[
                                    torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                            ###################################
                            if zero_mean:
                                z = z - z.mean(dim=1, keepdim=True)

                        if args.ilvr and t_ > args.t_thrs * args.diffusion_steps:
                            gamma_s = \
                                model.inflate_batch_array(model.gamma(t-1./args.diffusion_steps), x)
                            alpha_s = model.alpha(gamma_s, x)
                            sigma_s = model.sigma(gamma_s, x)
                            eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                                n_nodes=x.size(1),
                                node_mask=node_mask,
                                )
                            z_filtered = utils.mean_filter(z, K=50, radius=args.radius)
                            xs_filtered = x_filtered * alpha_s + sigma_s # utils.mean_filter(x * alpha_s + sigma_s * eps, K=50, radius=args.radius)
                            #xs_filtered = utils.mean_filter(x * alpha_s + sigma_s * eps, K=50, radius=args.radius)
                            z = z + args.ilvr * (- z_filtered + xs_filtered)
                        ###################################
                        if args.lambda_s > 0:
                            with torch.no_grad():
                                s_int = (t * args.diffusion_steps - 1).long()
                                domain_logit = domain_cls(z.permute(0,2,1),
                                        ts=s_int.flatten())['domain_cls']
                                #print(domain_logit.argmax(dim=1))
                        if guide: #args.egsde or args.entropy_guided:
                            y = model_dp(y, sample_p_zs_given_zt=True, s=t -
                                    1./args.diffusion_steps, t=t,
                                node_mask=node_mask, edge_mask=None, #fix_noise=False,
                                cond_fn=None, noise=noise) # w/o guidance for comparison

                        t = t - 1. / args.diffusion_steps

                    assert torch.all(t.abs() < 1e-5)
                if k == 0:
                    if False: #args.dataset.startswith('modelnet40') and not args.time_cond:
                        x_edit_list = [x_ori]
                    else:
                        x_edit_list = [alpha_t0 * x + noise_t0 * sigma_t0 if args.noise_t0 else x]
                    #x_edit_list_y = [alpha_t0 * x + noise_t0 * sigma_t0 if args.noise_t0 else x]
                    #labels_list = [labels]

                if guide: #args.egsde or args.entropy_guided:
                    x_edit_y = model_dp(y, sample_p_x_given_z0=True,
                            node_mask=node_mask, edge_mask=None,
                            #fix_noise=False,
                            ddim=args.ddim)
                    x_edit_list_y.append(x_edit_y)
                x_edit = model_dp(z, sample_p_x_given_z0=True,
                        node_mask=node_mask, edge_mask=None,
                        #fix_noise=False,
                        ddim=args.ddim)
                if 'aug' in args.mode:
                    pc_list.append(x_edit.detach().cpu())
                    label_list.append(labels.cpu())
                if args.accum_edit:
                    x = x_edit
                x_edit_list.append(x_edit)
                labels_list.append(labels)
                t = args.t * x.new_ones((x.shape[0], 1), device=x.device) # reset
            if args.K:
                x_edited_batch = torch.cat(x_edit_list[1:], dim=0).cpu()
                label_batch = torch.cat(labels_list[1:], dim=0).cpu()

                x_edited_batch_list.append(x_edited_batch)
                label_batch_list.append(label_batch)

            if args.cls_scale_mode == 'unit_norm':
                print("scaling")
                #x_edit_list = x_edit_list[0:1] + \
                x_edit_list = [rotate_shape_tensor(scale_to_unit_cube_torch(x.clone().detach()),
                        'x', np.pi/2) for x in
                            x_edit_list] # undo scaling
                #x_edit_list_y = x_edit_list_y[0:1] + \
                x_edit_list_y = [rotate_shape_tensor(scale_to_unit_cube_torch(x.clone().detach()),
                            'x', np.pi/2)
                                for x in x_edit_list_y] # undo scaling
            preds4vote = [] # hard
            preds4vote_y = [] # hard
            preds4vote_soft = []
            preds4vote_soft_y = []
            for k, x_edit in enumerate(x_edit_list): #, x_edit_list_y)):
                if args.time_cond:
                    logits = classifier(x_edit.permute(0, 2, 1),
                            activate_DefRec=False, ts=x_edit.new_zeros(len(x_edit)))
                else:
                    #if args.cls_scale_mode == 'unit_norm': #startswith('modelnet40'):
                    #    logits = classifier(rotate_shape_tensor(x_edit, 'x',
                    #        np.pi/2))
                    #else:
                    logits = classifier(x_edit)
                # ignored when (not classifier.time_cond)
                if guide: #args.egsde or args.entropy_guided:
                    x_edit_y = x_edit_list_y[k]
                    if args.time_cond:
                        logits_y = classifier(x_edit_y.permute(0, 2, 1),
                                activate_DefRec=False,
                                ts=x_edit_y.new_zeros(len(x_edit_y)))
                    else:
                        #if args.cls_scale_mode == 'unit_norm': #startswith('modelnet40'):
                        #    logits_y = classifier(rotate_shape_tensor(x_edit_y, 'x', np.pi/2))
                        #else:
                        logits_y = classifier(x_edit_y)
                    preds_y = logits_y["cls"].max(dim=1)[1]
                    correct_count_y[k] += (preds_y == labels).long().sum().item()
                preds = logits["cls"].max(dim=1)[1] # argmax
                if k == 0:
                    ori_preds = preds
                    ori_probs = logits["cls"].softmax(dim=-1)
                else:
                    preds4vote.append(preds)
                    preds4vote_soft.append(torch.softmax(logits['cls'], dim=-1))
                    if guide: #args.egsde or args.entropy_guided:
                        preds4vote_y.append(preds_y)
                        preds4vote_soft_y.append(torch.softmax(logits_y['cls'],
                            dim=-1))
                    #for ind in range(len(labels)):
                    #    pred_change[labels[ind].cpu(), ori_preds[ind].cpu(), preds[ind].cpu(), k-1] += 1

                correct_count[k] += (preds == labels).long().sum().item()
                if k>0:
                    correct_count_self_ensemble[k-1] += \
                        ((logits['cls'].softmax(dim=-1) +
                                ori_probs).max(dim=1)[1]  ==
                                labels).long().sum().item()
                    if args.ddim and args.time_cond: #ori_time_cond:
                        itmd_prob_k = []
                        itmd_pred_k = []
                        for xt, t in itmd_list[k-1]:
                            itmd_prob_k.append(torch.softmax(classifier(xt.permute(0,2,1),
                                    ts=xt.new_ones(len(xt))*t)['cls'], dim=-1))
                            itmd_pred_k.append(
                                itmd_prob_k[-1].argmax(dim=1))
                        correct_count_itmd[k-1][0] += (torch.stack(itmd_prob_k,
                                dim=1).mean(dim=1).argmax(dim=-1) ==
                                labels).long().sum().item() # soft voting of itmd
                        correct_count_itmd[k-1][1] += (torch.stack(itmd_pred_k,
                            dim=1).mode(dim=1).values ==
                            labels).long().sum().item() # hard voting of itmd

            io.cprint("ACC")
            for ck, cc in enumerate(correct_count):
                io.cprint(f'{ck} {cc/max(count, 1)*100}')
                if ck > 0:
                    if guide: #args.egsde or args.entropy_guided:
                        io.cprint(f'{ck} {correct_count_y[ck]/max(count,1)*100}')
                    io.cprint(f'self ensemble {correct_count_self_ensemble[ck-1] / max(count,1)*100}')
                    if args.ddim and ori_time_cond:
                        io.cprint(f'itmd voting soft {correct_count_itmd[ck-1][0] / max(count, 1)*100}')
                        io.cprint(f'itmd voting hard {correct_count_itmd[ck-1][1] / max(count, 1)*100}')

            if args.K > 1:
                preds_vote = torch.stack(preds4vote, dim=1).mode(dim=1).values
                probs_vote_soft =  torch.stack(preds4vote_soft,dim=1).mean(dim=1)
                preds_vote_soft = torch.stack(preds4vote_soft, dim=1).mean(dim=1).max(dim=1)[1]
                correct_count_vote += (preds_vote == labels).long().sum().item()
                correct_count_vote_soft += (preds_vote_soft == labels).long().sum().item()
                correct_count_vote_soft_ensemble += ((probs_vote_soft + ori_probs).max(dim=1)[1] ==
                        labels).long().sum().item()
                io.cprint(f'vote {correct_count_vote / max(count, 1)*100}')
                io.cprint(f'vote soft {correct_count_vote_soft / max(count, 1)*100}')
                io.cprint(f'vote soft ensemble {correct_count_vote_soft_ensemble / max(count, 1)*100}')

                if guide: #args.egsde or args.entropy_guided:
                    preds_vote_y = torch.stack(preds4vote_y, dim=1).mode(dim=1).values
                    probs_vote_soft_y =  torch.stack(preds4vote_soft_y,dim=1).mean(dim=1)
                    preds_vote_soft_y = torch.stack(preds4vote_soft_y, dim=1).mean(dim=1).max(dim=1)[1]
                    correct_count_vote_y += (preds_vote_y == labels).long().sum().item()
                    correct_count_vote_soft_y += (preds_vote_soft_y == labels).long().sum().item()
                    correct_count_vote_soft_ensemble_y += ((probs_vote_soft_y + ori_probs).max(dim=1)[1] ==
                            labels).long().sum().item()
                    io.cprint(f'vote (ori) {correct_count_vote_y / max(count, 1)*100}')
                    io.cprint(f'vote soft (ori) {correct_count_vote_soft_y / max(count, 1)*100}')
                    io.cprint(f'vote soft ensemble (ori) {correct_count_vote_soft_ensemble_y / max(count, 1)*100}')

        if args.K:
            x_edited_whole = torch.cat(x_edited_batch_list, dim=0)
            label_whole = torch.cat(label_batch_list, dim=0)
            #torch.save({'x': x_edited_whole, 'label':label_whole},
            #            f'data/shapenet_dm_aug_{args.t}.pt')
            #print('Save!', f'data/shapenet_dm_aug_{args.t}.pt')

        #for label, change in enumerate(pred_change):
        #    print(label)
        #    print(np.transpose(change, (2, 0, 1)))

        io.cprint(args)

        if 'aug' in args.mode:
            pcs = torch.cat(pc_list, dim=0)
            labels = torch.cat(label_list, dim=0)
            torch.save({'pcs': pcs, 'labels': labels},
                f"data/{args.dataset}_sdedit_{args.t}_ddim{args.ddim}_{args.n_reverse_steps}_lam_i{args.lambda_i}_{args.K}.pt")
            print("Save augmented data with DM!", "# instances: ", len(pcs))

    if 'vis' in args.mode:
      with torch.no_grad():
        for iter_idx, data in enumerate(test_loader_vis):
            labels = [idx_to_label[min(9, int(d))] for d in data[1]]
            print("GT", [idx_to_label[min(9, int(d))] for d in data[1]])
            x = data[0].to(device) #if not args.dataset.startswith('modelnet40') \
                                   #     else data[2].to(device)
            if args.pre_trans:
                mask = data[3].to(device)
                #x = pre_trans(x, mask)
                x = pre_trans_ver2(x, mask)
            rgbs = get_color(x.cpu().numpy())

            x_filtered = utils.mean_filter(x, K=50, radius=args.radius)
            rgbs_filtered = get_color(x_filtered.cpu().numpy())

            furthest_point_idx = \
                farthest_point_sample(x.permute(0,2,1), getattr(args,
                        'n_subsample', 64))[0]
            #furthest_point_idx = \
            #    pointnet2_utils.furthest_point_sample(x,
            #            getattr(args, 'n_subsample', 64))
            furthest_point_idx = furthest_point_idx.long()
            # x : batch_size x N x 3
            sub_x = \
                x[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]
            rgbs_sub = rgbs[np.arange(len(x)).reshape(-1, 1),
                    furthest_point_idx.cpu().numpy()]

            if args.cls_scale_mode == 'unit_norm':
                logits = \
                    classifier(scale_to_unit_cube_torch(x.clone().detach()))
            else:
                logits = classifier(x.clone().detach().permute(0, 2, 1),
                        ts=x.new_zeros(len(x)), activate_DefRec=False)
            preds = logits['cls'].max(dim=1)[1]
            prob_ori = logits['cls'].softmax(-1)
            preds_label = [idx_to_label[min(int(d), 9)] for d in preds]
            print("ori", preds_label)
            #record = (data[1].to(device) == preds).long().sum() / len(x) < 0.5
            #if not record:
            #    continue

            t = args.t * x.new_ones((x.shape[0], 1), device=x.device) # 0~1
            node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
            gamma_t = model.inflate_batch_array(model.gamma(t), x)
            alpha_t = model.alpha(gamma_t, x)
            sigma_t = model.sigma(gamma_t, x)

            @torch.enable_grad()
            def cond_fn(yt, t, phi, x, furthest_point_idx, model_kwargs):
                if args.lambda_ent == 0 and args.lambda_s == 0 and args.lambda_i == 0:
                    return torch.zeros_like(yt)
                # Ss (realistic expert)
                classifier.requires_grad_(True)
                model_dp.requires_grad_(True)
                yt.requires_grad_(True)
                #if args.model == 'pvd':
                #    print("!!!!!!!!!")
                #    model_dp.module.dynamics.train()
                #    model.train()
                #    model_dp.train()
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x).to(x)
                sigma_t = model.sigma(gamma_t, x).to(x)
                t_int = (t * model.T).long().float().flatten()

                entropy = 0
                domain_loss = 0
                #if args.lambda_s > 0:
                if args.lambda_ent>0:
                    if not ori_time_cond:
                        eps = phi(yt, t, **model_kwargs)
                        y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                        if args.cls_scale_mode == 'unit_norm':
                            y0_est_norm = scale_to_unit_cube_torch(y0_est)
                            cls_output = classifier(y0_est_norm.permute(0,2,1))
                        else:
                            cls_output = classifier(y0_est.permute(0,2,1),
                                    ts=torch.zeros_like(t).flatten())
                    else:
                        cls_output = classifier(yt.permute(0,2,1),
                                ts=t.flatten()*args.diffusion_steps)
                    cls_logits = cls_output['cls'] / args.temperature
                    cls_logits = cls_logits.to(x)
                    entropy = - (F.log_softmax(cls_logits, dim=-1) * F.softmax(cls_logits,
                        dim=-1)).sum(dim=-1).mean()
                if args.lambda_s > 0:
                    y_output = domain_cls(yt.permute(0,2,1), ts=t_int)
                    domain_y = y_output['domain_cls'].to(x)
                    feat_y = y_output['features'].permute(0,2,1)
                    domain_loss = F.cross_entropy(domain_y,
                            domain_y.new_zeros((len(domain_y),)).long())
                # Si (faithful expert)
                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=model_kwargs['node_mask'],
                    )
                if args.lambda_i > 0:
                    if args.voxelization:
                        if not args.latent_subdist: # in the sample space
                            eps = phi(yt, t, **model_kwargs)
                            x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                        furthest_point_idx]
                            y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                            vox_x, _, norm = voxelization(x.permute(0,2,1),
                                    x.permute(0,2,1))
                            vox_y = voxelization(y0_est.permute(0,2,1),
                                    y0_est.permute(0,2,1), norm=norm)[0]
                        else: # in the latent space
                            vox_x, _, norm = voxelization(xt.permute(0,2,1),
                                    xt.permute(0,2,1))
                            vox_y = voxelization(yt.permute(0,2,1),
                                    yt.permute(0,2,1), norm=norm)[0]
                        subdist_loss = F.mse_loss(vox_y, vox_x)
                    elif args.latent_subdist: # loss_i in latent space
                        xt = x * alpha_t + noise * sigma_t
                        xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        #subdist_loss = (xt_sub - yt_sub).pow(2).sum(-1).mean()
                        dist1, dist2, *_ = chamfer_dist_fn(xt_sub, yt)
                        #print(dist1.shape)
                        subdist_loss = dist1.mean()
                    else: # after model run
                        eps = phi(yt, t, **model_kwargs)
                        x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                        y0_est_sub = y0_est[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        #subdist_loss = (x_sub - y0_est_sub).pow(2).sum(-1).mean()
                        dist1_, dist2_, *_ = chamfer_dist_fn(x_sub, y0_est)
                        subdist_loss = dist1_.mean()
                else:
                    subdist_loss = 0
                grad = torch.autograd.grad(
                    (args.lambda_ent * entropy + args.lambda_s * domain_loss + args.lambda_i * subdist_loss),
                    yt, allow_unused=True)[0]
                #if args.model == 'pvd':
                #    model.eval()
                #    model_dp.eval()
                #    model_dp.module.dynamics.eval()
                return grad

            x_ori = x.detach()
            z_t_list = []
            x_edit_list = []
            x_edit_list_y = []
            preds_list = []
            preds_list_y = []
            for k in range(K):
                t = args.t * x.new_ones((x.shape[0], 1), device=x.device) # 0~1
                eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                z = alpha_t * x + sigma_t * eps # diffusion
                z_t_list.append(z)
                y = z.detach().clone()
                if args.dpm_solver:
                    model_fn = dpm_solver.model_wrapper(
                        model_dp,
                        ns,
                        model_kwargs={'cond_fn': cond_fn if guide else None, 'phi':True,
                            'node_mask': node_mask, 'x_ori': x,
                            'furthest_point_idx': furthest_point_idx,
                            },
                        guidance_type='uncond',
                        )
                    solver = dpm_solver.DPM_Solver(model_fn,
                            noise_schedule=ns,
                            algorithm_type='dpmsolver',
                            )
                    z = solver.sample(z, steps=args.n_reverse_steps, t_start=args.t,
                            order=3,
                            skip_type='time_uniform',
                            method='singlestep_fixed',
                            lower_order_final=True,
                            denoise_to_zero=False,
                            solver_type='dpmsolver',
                    )
                    # TODO keep_sub for dpm_solver

                    if guide:
                        model_fn = dpm_solver.model_wrapper(
                            model_dp,
                            ns,
                            model_kwargs={'cond_fn': None, 'phi':True,
                                'node_mask': node_mask, 'x_ori': x,
                                },
                            guidance_type='uncond',
                            )
                        solver = dpm_solver.DPM_Solver(model_fn,
                                noise_schedule=ns,
                                algorithm_type='dpmsolver',
                                )
                        y = solver.sample(y, steps=args.n_reverse_steps, t_start=args.t,
                                order=3,
                                skip_type='time_uniform',
                                method='singlestep_fixed',
                                lower_order_final=True,
                                denoise_to_zero=False,
                                solver_type='dpmsolver',
                        )

                elif args.ddim:
                    reverse_steps = np.linspace(args.diffusion_steps * args.t,
                            0, args.n_reverse_steps).astype(int) / args.diffusion_steps
                    print("Reverse Steps")
                    for t, s in tqdm(zip(reverse_steps[:-1], reverse_steps[1:]),
                            total=len(reverse_steps)-1):
                        t_tensor = t * x.new_ones((x.shape[0], 1), device=x.device)
                        s_tensor = s * x.new_ones((x.shape[0], 1), device=x.device)
                        z = model_dp(z, sample_p_zs_given_zt_ddim=True, s=s_tensor,
                                t=t_tensor, node_mask=node_mask, edge_mask=None, #None,
                                cond_fn=cond_fn if guide and t > args.t_thrs else None,
                                x_ori=x,
                                furthest_point_idx=furthest_point_idx,
                                )
                        if args.keep_sub:
                            gamma_s = \
                                    model.inflate_batch_array(model.gamma(s_tensor), x)
                            alpha_s = model.alpha(gamma_s, x)
                            sigma_s = model.sigma(gamma_s, x)
                            eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                                n_nodes=x.size(1),
                                node_mask=node_mask,
                                )
                            z[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                              furthest_point_idx] = (alpha_s * x + sigma_s * eps)[
                                    torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        if guide:
                            y = model_dp(y, sample_p_zs_given_zt_ddim=True, s=s_tensor,
                                    t=t_tensor, node_mask=node_mask, edge_mask=None, #None,
                                    )
                else:
                    for _ in tqdm(range(int(args.t * args.diffusion_steps))):
                        z, noise = model_dp(z, sample_p_zs_given_zt=True, s=t -
                                1./args.diffusion_steps, t=t,
                            node_mask=node_mask, edge_mask=None, #fix_noise=False,
                            cond_fn=cond_fn if guide else None,
                            x_ori=x,
                            furthest_point_idx=furthest_point_idx,
                            return_noise=True)
                        if args.keep_sub:
                            # for sub_x
                            gamma_s = \
                                model.inflate_batch_array(model.gamma(t-1./args.diffusion_steps), x)
                            alpha_s = model.alpha(gamma_s, x)
                            sigma_s = model.sigma(gamma_s, x)
                            eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                                n_nodes=x.size(1),
                                node_mask=node_mask,
                                )
                            z[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                              furthest_point_idx] = (alpha_s * x + sigma_s * eps)[
                                    torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        if guide:
                            y = model_dp(y, sample_p_zs_given_zt=True, s=t -
                                    1./args.diffusion_steps, t=t,
                                node_mask=node_mask, edge_mask=None, #fix_noise=False,
                                cond_fn=None, noise=noise) # w/o guidance for comparison
                        t = t - 1. / args.diffusion_steps
                    assert torch.all(t.abs() < 1e-5) #torch.allclose(t, torch.zeros_like(t))
                #for _ in tqdm(range(int(args.t * args.diffusion_steps))):
                #    z, noise = model_dp(z, sample_p_zs_given_zt=True, s=t -
                #            1./args.diffusion_steps, t=t,
                #        node_mask=node_mask, edge_mask=None, #None, #fix_noise=False,
                #        cond_fn=cond_fn_entropy if args.entropy_guided else (cond_fn_egsde if args.egsde else (cond_fn_sub
                #        if args.guidance_scale > 0 else None)), #noise=noise,
                #        x_ori=x,
                #        return_noise=True)
                #    if args.egsde or args.entropy_guided:
                #        y = model_dp(y, sample_p_zs_given_zt=True, s=t -
                #                1./args.diffusion_steps, t=t,
                #            node_mask=node_mask, edge_mask=None, #None, #fix_noise=False,
                #            cond_fn=None, noise=noise) #return_noise=True)
                #    t = t - 1. / args.diffusion_steps
                #    print('diff', (z-y).abs().max(), (z-y).abs().mean())
                #assert torch.all(t.abs() < 1e-5) #torch.allclose(t, torch.zeros_like(t))
                x_edit = model_dp(z, sample_p_x_given_z0=True,
                        node_mask=node_mask, edge_mask=None, #None,
                        ddim=args.ddim,
                        #fix_noise=False
                        )
                x = x_edit

                if guide: #args.egsde or args.entropy_guided:
                    x_edit_y = model_dp(y, sample_p_x_given_z0=True,
                            node_mask=node_mask, edge_mask=None,
                            ddim=args.ddim,
                            #None,
                            #fix_noise=False
                            )
                    y = x_edit_y

                if args.cls_scale_mode == 'unit_norm':
                    logits = \
                        classifier(scale_to_unit_cube_torch(x_edit.clone().detach()))
                        #.permute(0, 2, 1), activate_DefRec=False)
                        #classifier(x_edit.clone().detach().permute(0, 2, 1), activate_DefRec=False)
                elif args.cls_scale_mode == 'unit_std':
                    logits = \
                        classifier(x_edit.clone().detach().permute(0, 2, 1),
                                ts=x_edit.new_zeros(len(x_edit)), activate_DefRec=False)


                preds = logits["cls"].max(dim=1)[1]
                preds_val = logits["cls"].softmax(-1).max(dim=1)[0]
                preds_label = [idx_to_label[min(9, int(d))] for d in preds]

                print(k, preds_label)
                preds_list.append(zip(preds_label, preds_val))
                #x = x - x.mean(dim=1, keepdim=True)
                #norm = x.pow(2).sum(-1).sqrt().max(-1).values
                #x /= norm[:, None, None]
                x_edit_list.append(x_edit)
                t = args.t * x.new_ones((x.shape[0], 1), device=x.device) # reset

                if guide: #args.egsde or args.entropy_guided:
                    if args.cls_scale_mode == 'unit_norm':
                        logits_y = \
                            classifier(scale_to_unit_cube_torch(x_edit_y.clone().detach()))#.permute(0, 2, 1), activate_DefRec=False)
                            #classifier(x_edit.clone().detach().permute(0, 2, 1), activate_DefRec=False)
                    else:
                        logits_y = \
                            classifier(x_edit_y.clone().detach().permute(0, 2,
                                1), ts=x_edit_y.new_zeros(len(x_edit_y)), activate_DefRec=False)
                    preds_y = logits_y["cls"].max(dim=1)[1]
                    preds_val_y = logits_y["cls"].softmax(-1).max(dim=1)[0]
                    preds_label_y = [idx_to_label[min(9, int(d))] for d in preds_y]
                    print("GUIDED", 'lambda_ent', args.lambda_ent, 'lambda_s',
                            args.lambda_s, 'lamba_i', args.lambda_i)
                    print(k, preds_label_y)
                    preds_list_y.append(zip(preds_label_y, preds_val_y))
                    x_edit_list_y.append(x_edit_y)
            if args.cls_scale_mode == 'unit_norm':
                logits = \
                    classifier(scale_to_unit_cube_torch(x_ori.clone().detach()))#.permute(0, 2, 1), activate_DefRec=False)
            else:
                logits = classifier(x_ori.clone().detach().permute(0, 2, 1),
                        ts=x_ori.new_zeros(len(x_ori)), activate_DefRec=False)
            preds = logits["cls"].max(dim=1)[1]
            preds_val = logits["cls"].softmax(-1).max(dim=1)[0]
            preds_label = [idx_to_label[min(9, int(d))] for d in preds]
            #for b, (pred_label, pred_val)  in enumerate(zip(preds_label, preds_val)):
            for iter, (z_t, x_edit, preds_edit) in enumerate(zip(z_t_list, x_edit_list, preds_list)):
                rgbs_edit = get_color(x_edit.cpu().numpy())

                if guide: #args.egsde or args.entropy_guided:
                    x_edit_y = x_edit_list_y[iter]
                    preds_edit_y = preds_list_y[iter]
                    rgbs_edit = get_color(x_edit_y.cpu().numpy())
                    preds_edit_y = [a for a in preds_edit_y]

                for b, (pred_label_ori, pred_val_ori, (pred_label, pred_val)) in enumerate(zip(preds_label,
                    preds_val, preds_edit)): #range(args.batch_size):
                    if iter == 0:
                        obj3d = wandb.Object3D({
                            "type": "lidar/beta",
                            "points": np.concatenate((x_ori[b].cpu().numpy().reshape(-1, 3),
                                rgbs[b]), axis=1),
                            "boxes": np.array(
                                [
                                    {
                                        "corners":
                                        (np.array([
                                            [-1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [1, -1, -1],
                                            [1, 1, -1],
                                            [-1, 1, 1],
                                            [1, -1, 1],
                                            [1, 1, 1]
                                        ])*args.scale * (1 if args.scale_mode !=
                                        'unit_std' else 3)).tolist(),
                                        "label": f'{labels[b]} {pred_label_ori} {pred_val_ori:.2f}',
                                        "color": [123, 321, 111], # ???
                                    }
                                ]
                            ),
                        })
                        wandb.log({f'ori': obj3d}, step=b, commit=False)

                        obj3d = wandb.Object3D({
                            "type": "lidar/beta",
                            "points": np.concatenate((x_filtered[b].cpu().numpy().reshape(-1, 3),
                                rgbs_filtered[b]), axis=1),
                            "boxes": np.array(
                                [
                                    {
                                        "corners":
                                        (np.array([
                                            [-1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [1, -1, -1],
                                            [1, 1, -1],
                                            [-1, 1, 1],
                                            [1, -1, 1],
                                            [1, 1, 1]
                                        ])*args.scale * (1 if args.scale_mode !=
                                        'unit_std' else 3)).tolist(),
                                        "label": f'{labels[b]} {pred_label_ori} {pred_val_ori:.2f}',
                                        "color": [123, 321, 111], # ???
                                    }
                                ]
                            ),
                        })
                        wandb.log({f'filtered': obj3d}, step=b, commit=False)

                        obj3d = wandb.Object3D({
                            "type": "lidar/beta",
                            "points": np.concatenate((sub_x[b].cpu().numpy().reshape(-1, 3),
                                rgbs_sub[b]), axis=1),
                            "boxes": np.array(
                                [
                                    {
                                        "corners":
                                        (np.array([
                                            [-1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [1, -1, -1],
                                            [1, 1, -1],
                                            [-1, 1, 1],
                                            [1, -1, 1],
                                            [1, 1, 1]
                                        ])*args.scale * (1 if args.scale_mode !=
                                        'unit_std' else 3)).tolist(),
                                        "label": f'{labels[b]} {pred_label_ori} {pred_val_ori:.2f}',
                                        "color": [123, 321, 111], # ???
                                    }
                                ]
                            ),
                        })
                        wandb.log({f'ori sub': obj3d}, step=b, commit=False)

                    obj3d = wandb.Object3D({
                        "type": "lidar/beta",
                        "points": np.concatenate((z_t[b].cpu().numpy().reshape(-1, 3),
                            rgbs[b]), axis=1),
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
                                    ])*args.scale * (1 if args.scale_mode !=
                                    'unit_std' else 3)).tolist(),
                                    "label": f'{labels[b]} {pred_label} {pred_val:.2f}',
                                    "color": [123, 321, 111], # ???
                                }
                            ]
                        ),
                    })
                    wandb.log({f'corrupted {iter}': obj3d}, step=b, commit=False)


                    obj3d = wandb.Object3D({
                        "type": "lidar/beta",
                        "points": np.concatenate((x_edit[b].cpu().numpy().reshape(-1, 3),
                            rgbs[b]), axis=1),
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
                                    ])*args.scale*(1 if args.scale_mode !=
                                    'unit_std' else 3)).tolist(),
                                    "label": f'{labels[b]} {pred_label} {pred_val:.2f}',
                                    "color": [123, 321, 111], # ???
                                }
                            ]
                        ),
                    })
                    wandb.log({f'edit {iter}': obj3d}, step=b, commit=False)

                    obj3d = wandb.Object3D({
                        "type": "lidar/beta",
                        "points": np.concatenate((x_edit[b].cpu().numpy().reshape(-1, 3),
                            rgbs_edit[b]), axis=1),
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
                                    ])*args.scale*(1 if args.scale_mode !=
                                    'unit_std' else 3)).tolist(),
                                    "label": f'{labels[b]} {pred_label} {pred_val:.2f}',
                                    "color": [123, 321, 111], # ???
                                }
                            ]
                        ),
                    })
                    wandb.log({f'edit {iter} re-colored': obj3d}, step=b, commit=False)

                    if guide: #args.egsde or args.entropy_guided:
                        pred_label, pred_val = preds_edit_y[b]

                        obj3d = wandb.Object3D({
                            "type": "lidar/beta",
                            "points": np.concatenate((x_edit_y[b].cpu().numpy().reshape(-1, 3),
                                rgbs[b]), axis=1),
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
                                        ])*args.scale*(1 if args.scale_mode !=
                                        'unit_std' else 3)).tolist(),
                                        "label": f'{labels[b]} {pred_label} {pred_val:.2f}',
                                        "color": [123, 321, 111], # ???
                                    }
                                ]
                            ),
                        })
                        wandb.log({f'edit_y {iter}': obj3d}, step=b, commit=False)

                        obj3d = wandb.Object3D({
                            "type": "lidar/beta",
                            "points": np.concatenate((x_edit_y[b].cpu().numpy().reshape(-1, 3),
                                rgbs_edit[b]), axis=1),
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
                                        ])*args.scale*(1 if args.scale_mode !=
                                        'unit_std' else 3)).tolist(),
                                        "label": f'{labels[b]} {pred_label} {pred_val:.2f}',
                                        "color": [123, 321, 111], # ???
                                    }
                                ]
                            ),
                        })
                        wandb.log({f'edit y {iter} re-colored': obj3d}, step=b, commit=False)
            break
    print("Visualization Done")

if __name__ == "__main__":
    main()
