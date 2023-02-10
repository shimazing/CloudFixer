# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
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

from data.dataloader_Norm import ScanNet, ModelNet, ShapeNet, label_to_idx, NUM_POINTS
from data.dataloader_Norm import ShapeNetCore
from data.dataloader_Norm import idx_to_label
from utils_GAST import pc_utils_Norm, log
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from Models_Norm import PointNet, DGCNN
from utils_GAST.pc_utils_Norm import scale_to_unit_cube_torch
from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F
from voxelization_guide import Voxelization

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
        if dataset.partition == 'train':
            return dataset.label[dataset.train_ind]
        elif dataset.partition == 'val':
            return dataset.label[dataset.val_ind]
        return dataset.label

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
#parser.add_argument('--lr', type=float, default=1e-4) #2e-4)
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
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--voting', type=str, default='hard', choices=['hard', 'soft'])
parser.add_argument('--accum_edit', action='store_true', help='Disable wandb') # TODO
parser.add_argument('--scale_mode', type=str, default='unit_std')
parser.add_argument('--cls_scale_mode', type=str, default='unit_norm')
parser.add_argument('--scale', type=float, default=3)
########################
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--output_pts', type=int, default=512)

######################## guided sampling
parser.add_argument('--guidance_scale', type=float, default=0)
parser.add_argument('--mode', nargs='+', type=str, default=['eval'])
parser.add_argument('--dataset', type=str, default='shapenet')
parser.add_argument('--keep_sub', action='store_true', help='Disable wandb') # TODO
parser.add_argument('--n_subsample', type=int, default=64)
parser.add_argument('--classifier', type=str,
    default='../GAST/experiments/GAST_balanced_unitnorm_randomscale0.2/model.ptdgcnn')
parser.add_argument('--self_ensemble', action='store_true')
parser.add_argument('--egsde', action='store_true')
parser.add_argument('--domain_cls', type=str, default='outputs/domain_classifier_DGCNN_shape_model_timecondGN.pt')
parser.add_argument('--entropy_guided', action='store_true')
parser.add_argument('--lambda_s', default=100, type=float)
parser.add_argument('--lambda_i', default=1, type=float)
parser.add_argument('--random_seed', default=0, type=int)

parser.add_argument('--ddim', action='store_true')
parser.add_argument('--n_inversion_steps', type=int, default=50) #action='store_true')
parser.add_argument('--n_reverse_steps', type=int, default=50) #action='store_true')
parser.add_argument('--voxel_resolution', type=int, default=32)
parser.add_argument('--voxelization', action='store_true')

args = parser.parse_args()


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


if args.n_nodes == 1024:
    dataset_ = ShapeNet(io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=True)
    if args.dataset == 'scannet':
        test_dataset = ScanNet(io, './data', 'test', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=False) # for classification
    elif args.dataset == 'modelnet':
        test_dataset = ModelNet(io, './data', 'test', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=False) # for classification
    elif args.dataset == 'shapenet':
        test_dataset = ShapeNet(io, './data', 'test', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_rotation=False) # for classification
    # TODO jitter??!!
    train_dataset_sampler, val_dataset_sampler = split_set(dataset_,
        domain='shapenet')

    train_loader = DataLoader(dataset_, batch_size=args.batch_size,
            sampler=train_dataset_sampler,
            drop_last=False)
    val_loader = DataLoader(dataset_, batch_size=args.batch_size,
            sampler=val_dataset_sampler) #, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False)
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

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.ddim:
    # deterministic
    args.K = 1

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
    print(args)

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
model = model.to(device)


ori_args_model = args.model

args.model = 'dgcnn'
args.time_cond = False
classifier = DGCNN(args).to(device).eval()

classifier.load_state_dict(
torch.load(
    args.classifier, map_location='cpu'),
strict=False
)

args.model = ori_args_model


if args.egsde:
    from chamfer_distance import ChamferDistance as chamfer_dist
    chamfer_dist_fn = chamfer_dist()
    args.time_cond = True
    domain_cls = DGCNN(args).to(device).eval()
    domain_cls_state_dict = torch.load(
        args.domain_cls,
        map_location='cpu')
    keys = list(domain_cls_state_dict.keys())
    for key in keys:
        if 'input_transform_net' in key:
            domain_cls_state_dict.pop(key)
    domain_cls.load_state_dict(
        domain_cls_state_dict, strict=False
        )
if args.entropy_guided:
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


def main():
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    model.eval()
    K = args.K

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    if 'eval' in args.mode: # calc acc
      with torch.no_grad():
        count = 0
        correct_count = [0 for _ in range(K+1)]
        correct_count_y = [0 for _ in range(K+1)]
        correct_count_self_ensemble = [0 for _ in range(K)]
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
            labels = data[1].to(device)
            print(labels)
            x = data[0].to(device)

            furthest_point_idx = \
                pointnet2_utils.furthest_point_sample(x,
                        getattr(args, 'n_subsample', 64))
            furthest_point_idx = furthest_point_idx.long()
            sub_x = \
                x[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]

            count += len(x)
            t = args.t * x.new_ones((x.shape[0], 1), device=x.device)
            node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
            gamma_t = model.inflate_batch_array(model.gamma(t), x)
            alpha_t = model.alpha(gamma_t, x)
            sigma_t = model.sigma(gamma_t, x)

            @torch.enable_grad()
            def cond_fn_entropy(yt, t, phi, model_kwargs):
                # Ss (realistic expert)
                classifier.requires_grad_(True)
                yt.requires_grad_(True)
                if args.model == 'pvd':
                    model.train()
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)
                if args.lambda_s > 0:
                    eps = phi(yt, t, **model_kwargs)
                    y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                    y0_est_norm = scale_to_unit_cube_torch(y0_est)
                    cls_output = classifier(y0_est_norm.permute(0,2,1))
                    cls_logits = cls_output['cls']
                    entropy = - (F.log_softmax(cls_logits, dim=-1) * F.softmax(cls_logits,
                        dim=-1)).sum(dim=-1).mean()
                else:
                    entropy = 0
                # Si (faithful expert)
                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                if args.lambda_i > 0:
                    if False: # loss_i in latent space
                        xt = x * alpha_t + noise * sigma_t
                        xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        subdist_loss = (xt_sub - yt_sub).pow(2).sum(-1).mean()
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
                        subdist_loss = (x_sub - y0_est_sub).pow(2).sum(-1).mean()
                        dist1_, dist2_, *_ = chamfer_dist_fn(x_sub, y0_est)
                        subdist_loss = dist1_.mean()
                    print(subdist_loss)
                else:
                    subdist_loss = 0
                grad = torch.autograd.grad(
                    (args.lambda_s * entropy + args.lambda_i * subdist_loss),
                    yt, allow_unused=True)[0]
                if args.model == 'pvd':
                    model.eval()
                return grad

            @torch.enable_grad()
            def cond_fn_egsde(yt, t, phi, model_kwargs):
                if args.model == 'pvd':
                    model.train()
                yt.requires_grad_(True)
                # Ss (realistic expert)
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)
                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                xt = x * alpha_t + noise * sigma_t
                t_int = (t * model.T).long().float().flatten()

                x_output = domain_cls(xt.permute(0,2,1), ts=t_int)
                domain_x = x_output['domain_cls']
                feat_x = x_output['features'].permute(0,2,1)

                y_output = domain_cls(yt.permute(0,2,1), ts=t_int)
                domain_y = y_output['domain_cls']
                feat_y = y_output['features'].permute(0,2,1)
                if True:
                    if args.lambda_s > 0:
                        domain_loss = F.cross_entropy(domain_y,
                                domain_y.new_zeros((len(domain_y),)).long())
                    else:
                        domain_loss = 0
                else:
                    domain_loss = F.cosine_similarity(feat_x, feat_y, dim=-1).mean()

                # Si (faithful expert)
                if args.lambda_i > 0:
                    if args.voxelization:
                        if False: # in the sample space
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
                    elif False: # chamfer dist in the latent space
                        xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        subdist_loss = (xt_sub - yt_sub).pow(2).sum(-1).mean()
                        dist1, dist2, *_ = chamfer_dist_fn(xt_sub, yt)
                    else: # chamfer dist in the sample space
                        eps = phi(yt, t, **model_kwargs)
                        x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                        dist1_, dist2_, *_ = chamfer_dist_fn(x_sub, y0_est)
                        subdist_loss = dist1_.mean()
                else:
                    subdist_loss = 0
                grad = torch.autograd.grad(
                    (args.lambda_s * domain_loss + args.lambda_i * subdist_loss),
                    yt, allow_unused=True)[0]
                if args.model == 'pvd':
                    model.eval()
                return grad

            @torch.enable_grad()
            def cond_fn_sub_naive(yt, t, phi, model_kwargs):
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)
                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                xt = x * alpha_t + noise * sigma_t
                xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]
                yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]

                loss = (xt_sub - yt_sub).pow(2).mean()
                grad = torch.autograd.grad(loss, yt, allow_unused=True)[0]
                return grad


            @torch.enable_grad()
            def cond_fn_sub(xt, t, phi, model_kwargs):
                model.dynamics.requires_grad_(True)

                #sub_x = \
                #    pointnet2_utils.gather_operation(x.transpose(1,2).contiguous(), furthest_point_idx).transpose(1, 2).contiguous()

                xt.requires_grad_(True)
                eps = phi(xt, t, **model_kwargs)
                # batch_size x N x 3

                gamma_t = model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)

                x0_est = 1 / alpha_t * (xt  - sigma_t * eps)
                sub_x0_est = x0_est[torch.arange(len(x)).view(-1,
                    1).to(furthest_point_idx),
                        furthest_point_idx]
                #sub_x0_est = \
                #    pointnet2_utils.gather_operation(x0_est.transpose(1,2).contiguous(), furthest_point_idx).transpose(1, 2).contiguous()

                loss = args.guidance_scale * (sub_x0_est - sub_x).pow(2).mean()

                grad = torch.autograd.grad(loss, xt, allow_unused=True)[0]
                return grad


            #x_ori = x
            x_edit_list = [x]
            x_edit_list_y = [x]
            labels_list = [labels]
            for k in range(K):
                print(k+1,  "/", K)
                eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                if args.ddim:
                    inversion_steps = np.linspace(0, args.t, args.n_inversion_steps)
                    # inversion
                    z = x.detach().clone()
                    print("Inversion")
                    for t, s in tqdm(zip(inversion_steps[:-1],
                        inversion_steps[1:]), total=len(inversion_steps)-1):
                        t_tensor = t * x.new_ones((x.shape[0], 1), device=x.device)
                        s_tensor = s * x.new_ones((x.shape[0], 1), device=x.device)
                        z = model.sample_p_zs_given_zt_ddim(s_tensor,
                                t_tensor, z, node_mask, edge_mask=None, cond_fn=None)
                else:
                    z = alpha_t * x + sigma_t * eps # diffusion

                y = z.detach().clone()
                if args.ddim:
                    reverse_steps = np.linspace(args.t, 0, args.n_reverse_steps)
                    print("Reverse Steps")
                    for t, s in tqdm(zip(reverse_steps[:-1], reverse_steps[1:]),
                            total=len(reverse_steps)-1):
                        t_tensor = t * x.new_ones((x.shape[0], 1), device=x.device)
                        s_tensor = s * x.new_ones((x.shape[0], 1), device=x.device)
                        z = model.sample_p_zs_given_zt_ddim(s_tensor,
                                t_tensor, z, node_mask, None, #None,
                                cond_fn=cond_fn_entropy if args.entropy_guided else (cond_fn_egsde if args.egsde else (cond_fn_sub
                                if args.guidance_scale > 0 else None)), #noise=noise,
                                )
                        if args.egsde or args.entropy_guided:
                            y = model.sample_p_zs_given_zt_ddim(s_tensor,
                                    t_tensor, y, node_mask, None, #None,
                                    )
                else:
                    for _ in tqdm(range(int(args.t * args.diffusion_steps))):
                        z, noise = model.sample_p_zs_given_zt(t - 1./args.diffusion_steps, t, z,
                            node_mask, edge_mask=None, fix_noise=False,
                            cond_fn=cond_fn_entropy if args.entropy_guided else (cond_fn_egsde if args.egsde else (cond_fn_sub
                            if args.guidance_scale > 0 else None)),
                            return_noise=True)
                        if args.egsde or args.entropy_guided:
                            y = model.sample_p_zs_given_zt(t - 1./args.diffusion_steps, t, y,
                                node_mask, edge_mask=None, fix_noise=False,
                                cond_fn=None, noise=noise) # w/o guidance for comparison


                        # constraint (like ilvr)
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
                            ###################################
                            z = z - z.mean(dim=1, keepdim=True)
                            ###################################
                        t = t - 1. / args.diffusion_steps

                    assert torch.all(t.abs() < 1e-5)

                if args.egsde or args.entropy_guided:
                    x_edit_y = model.sample_p_x_given_z0(y, node_mask, None,
                            fix_noise=False, ddim=args.ddim)
                    x_edit_list_y.append(x_edit_y)
                x_edit = model.sample_p_x_given_z0(z, node_mask, None,
                        fix_noise=False, ddim=args.ddim)
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
                x_edit_list = [scale_to_unit_cube_torch(x.clone().detach()) for x in x_edit_list] # undo scaling
                x_edit_list_y = [scale_to_unit_cube_torch(x.clone().detach())
                        for x in x_edit_list_y] # undo scaling
            preds4vote = [] # hard
            preds4vote_y = [] # hard
            preds4vote_soft = []
            preds4vote_soft_y = []
            for k, x_edit in enumerate(x_edit_list): #, x_edit_list_y)):
                logits = classifier(x_edit.permute(0, 2, 1), activate_DefRec=False)
                if args.egsde or args.entropy_guided:
                    x_edit_y = x_edit_list_y[k]
                    logits_y = classifier(x_edit_y.permute(0, 2, 1), activate_DefRec=False)
                    preds_y = logits_y["cls"].max(dim=1)[1]
                    correct_count_y[k] += (preds_y == labels).long().sum().item()
                preds = logits["cls"].max(dim=1)[1] # argmax
                if k == 0:
                    ori_preds = preds
                    ori_probs = logits["cls"].softmax(dim=-1)
                else:
                    preds4vote.append(preds)
                    preds4vote_soft.append(torch.softmax(logits['cls'], dim=-1))
                    if args.egsde or args.entropy_guided:
                        preds4vote_y.append(preds_y)
                        preds4vote_soft_y.append(torch.softmax(logits_y['cls'],
                            dim=-1))
                    for ind in range(len(labels)):
                        pred_change[labels[ind].cpu(), ori_preds[ind].cpu(), preds[ind].cpu(), k-1] += 1

                    #for label, ori, pred in zip(labels, ori_preds, preds):
                    #    pred_change[int(label)][int(ori), int(pred), k-1] += 1
                correct_count[k] += (preds == labels).long().sum().item()
                if k>0:
                    correct_count_self_ensemble[k-1] += \
                        ((logits['cls'].softmax(dim=-1) +
                                ori_probs).max(dim=1)[1]  ==
                                labels).long().sum().item()


            print("ACC")
            for ck, cc in enumerate(correct_count):
                print(ck, cc/max(count, 1)*100)
                if ck > 0:
                    if args.egsde or args.entropy_guided:
                        print(ck, correct_count_y[ck]/max(count, 1)*100)
                    print('self ensemble', correct_count_self_ensemble[ck-1] /
                            max(count,1)*100)

            if args.K > 1:
                preds_vote = torch.stack(preds4vote, dim=1).mode(dim=1).values
                probs_vote_soft =  torch.stack(preds4vote_soft,dim=1).mean(dim=1)
                preds_vote_soft = torch.stack(preds4vote_soft, dim=1).mean(dim=1).max(dim=1)[1]
                correct_count_vote += (preds_vote == labels).long().sum().item()
                correct_count_vote_soft += (preds_vote_soft == labels).long().sum().item()
                correct_count_vote_soft_ensemble += ((probs_vote_soft + ori_probs).max(dim=1)[1] ==
                        labels).long().sum().item()
                print('vote', correct_count_vote / max(count, 1)*100)
                print('vote soft', correct_count_vote_soft / max(count, 1)*100)
                print('vote soft ensemble', correct_count_vote_soft_ensemble / max(count, 1)*100)

                if args.egsde or args.entropy_guided:
                    preds_vote_y = torch.stack(preds4vote_y, dim=1).mode(dim=1).values
                    probs_vote_soft_y =  torch.stack(preds4vote_soft_y,dim=1).mean(dim=1)
                    preds_vote_soft_y = torch.stack(preds4vote_soft_y, dim=1).mean(dim=1).max(dim=1)[1]
                    correct_count_vote_y += (preds_vote_y == labels).long().sum().item()
                    correct_count_vote_soft_y += (preds_vote_soft_y == labels).long().sum().item()
                    correct_count_vote_soft_ensemble_y += ((probs_vote_soft_y + ori_probs).max(dim=1)[1] ==
                            labels).long().sum().item()
                    print('vote (ori)', correct_count_vote_y / max(count, 1)*100)
                    print('vote soft (ori)', correct_count_vote_soft_y / max(count, 1)*100)
                    print('vote soft ensemble (ori)',
                            correct_count_vote_soft_ensemble_y / max(count, 1)*100)


        if args.K:
            x_edited_whole = torch.cat(x_edited_batch_list, dim=0)
            label_whole = torch.cat(label_batch_list, dim=0)
            #torch.save({'x': x_edited_whole, 'label':label_whole},
            #            f'data/shapenet_dm_aug_{args.t}.pt')
            #print('Save!', f'data/shapenet_dm_aug_{args.t}.pt')

        for label, change in enumerate(pred_change):
            print(label)
            print(np.transpose(change, (2, 0, 1)))

        print(args)

    if 'vis' in args.mode:
      with torch.no_grad():
        for iter_idx, data in enumerate(test_loader_vis):
            labels = [idx_to_label[int(d)] for d in data[1]]

            print([idx_to_label[int(d)] for d in data[1]])
            x = data[0].to(device)
            rgbs = get_color(x.cpu().numpy())

            furthest_point_idx = \
                pointnet2_utils.furthest_point_sample(x,
                        getattr(args, 'n_subsample', 64))
            furthest_point_idx = furthest_point_idx.long()
            #print(furthest_point_idx)
            #input()
            # x : batch_size x N x 3
            sub_x = \
                x[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]
            rgbs_sub = rgbs[np.arange(len(x)).reshape(-1, 1),
                    furthest_point_idx.cpu().numpy()]

            #logits = classifier(scale_to_unit_cube_torch(x.clone().detach()).permute(0, 2, 1), activate_DefRec=False)
            logits = classifier(x.clone().detach().permute(0, 2, 1), activate_DefRec=False)
            preds = logits['cls'].max(dim=1)[1]
            prob_ori = logits['cls'].softmax(-1)
            preds_label = [idx_to_label[int(d)] for d in preds]
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
            def cond_fn_entropy(yt, t, phi, model_kwargs):
                # Ss (realistic expert)
                classifier.requires_grad_(True)
                yt.requires_grad_(True)
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)
                if True: #args.lambda_s > 0:
                    eps = phi(yt, t, **model_kwargs)
                    y0_est = 1 / alpha_t * (yt  - sigma_t * eps)

                    y0_est_norm = scale_to_unit_cube_torch(y0_est)

                    cls_output = classifier(y0_est_norm.permute(0,2,1))
                    cls_logits = cls_output['cls']

                    entropy = - (F.log_softmax(cls_logits, dim=-1) * F.softmax(cls_logits,
                        dim=-1)).sum(dim=-1).mean()
                else:
                    entropy = 0

                # Si (faithful expert)

                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                if args.lambda_i > 0:
                    if False:
                        xt = x * alpha_t + noise * sigma_t
                        xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        subdist_loss = (xt_sub - yt_sub).pow(2).sum(-1).mean()
                        dist1, dist2, *_ = chamfer_dist_fn(xt_sub, yt)
                        #print(dist1.shape)
                        subdist_loss = dist1.mean()
                    else:
                        eps = phi(yt, t, **model_kwargs)
                        x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                        y0_est_sub = y0_est[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        subdist_loss = (x_sub - y0_est_sub).pow(2).sum(-1).mean()
                        dist1_, dist2_, *_ = chamfer_dist_fn(x_sub, y0_est)
                        subdist_loss = dist1_.mean()
                    print(subdist_loss)
                else:
                    subdist_loss = 0
                grad = torch.autograd.grad(
                    (args.lambda_s * entropy + args.lambda_i * subdist_loss),
                    yt, allow_unused=True)[0]
                #print(grad)
                #print(grad[grad.nonzero(as_tuple=True)])
                #print("!!!!!!!!!!!", len(grad.nonzero(as_tuple=True)[0]),
                #"!!!!!!!!")
                return grad

            @torch.enable_grad()
            def cond_fn_egsde(yt, t, phi, model_kwargs):
                if args.model == 'pvd':
                    model.train()
                yt.requires_grad_(True)
                # Ss (realistic expert)
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)
                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                xt = x * alpha_t + noise * sigma_t
                t_int = (t * model.T).long().float().flatten()

                x_output = domain_cls(xt.permute(0,2,1), ts=t_int)
                domain_x = x_output['domain_cls']
                feat_x = x_output['features'].permute(0,2,1)

                y_output = domain_cls(yt.permute(0,2,1), ts=t_int)
                domain_y = y_output['domain_cls']
                print(domain_y.argmax(dim=1), "!!!!!!!!!!!!")
                feat_y = y_output['features'].permute(0,2,1)
                if True:
                    if args.lambda_s > 0:
                        domain_loss = F.cross_entropy(domain_y,
                                domain_y.new_zeros((len(domain_y),)).long())
                    else:
                        domain_loss = 0
                else:
                    domain_loss = F.cosine_similarity(feat_x, feat_y, dim=-1).mean()

                # Si (faithful expert)

                if args.lambda_i > 0:
                    if args.voxelization:
                        if False:
                            eps = phi(yt, t, **model_kwargs)
                            x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                        furthest_point_idx]
                            y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                            vox_x, _, norm = voxelization(x.permute(0,2,1),
                                    x.permute(0,2,1))
                            vox_y = voxelization(y0_est.permute(0,2,1),
                                    y0_est.permute(0,2,1), norm=norm)[0]
                        else:
                            vox_x, _, norm = voxelization(xt.permute(0,2,1),
                                    xt.permute(0,2,1))
                            vox_y = voxelization(yt.permute(0,2,1),
                                    yt.permute(0,2,1), norm=norm)[0]
                        subdist_loss = F.mse_loss(vox_y, vox_x)
                    elif False:
                        xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
                                furthest_point_idx]
                        subdist_loss = (xt_sub - yt_sub).pow(2).sum(-1).mean()
                        dist1, dist2, *_ = chamfer_dist_fn(xt_sub, yt)
                        #print(dist1.shape)
                        #subdist_loss = dist1.mean()
                    else:
                        eps = phi(yt, t, **model_kwargs)
                        x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
                                    furthest_point_idx]
                        y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
                        dist1_, dist2_, *_ = chamfer_dist_fn(x_sub, y0_est)
                        subdist_loss = dist1_.mean()
                else:
                    subdist_loss = 0
                grad = torch.autograd.grad(
                    (args.lambda_s * domain_loss + args.lambda_i * subdist_loss),
                    yt, allow_unused=True)[0]
                #print(grad.min(), grad.max(), grad.abs().mean()) #[grad.nonzero(as_tuple=True)])
                #print("!!!!!!!!!!!", len(grad.nonzero(as_tuple=True)[0]),
                #        grad.shape)
                if args.model == 'pvd':
                    model.eval()
                return grad

            #@torch.enable_grad()
            #def cond_fn_egsde(yt, t, phi, model_kwargs):
            #    if args.model == 'pvd':
            #        model.train()
            #    yt.requires_grad_(True)
            #    # Ss (realistic expert)
            #    gamma_t = \
            #        model.inflate_batch_array(model.gamma(t), x)
            #    alpha_t = model.alpha(gamma_t, x)
            #    sigma_t = model.sigma(gamma_t, x)
            #    noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
            #        n_nodes=x.size(1),
            #        node_mask=node_mask,
            #        )
            #    xt = x * alpha_t + noise * sigma_t
            #    t_int = (t * model.T).long().float().flatten()

            #    x_output = domain_cls(xt.permute(0,2,1), ts=t_int)
            #    domain_x = x_output['domain_cls']
            #    feat_x = x_output['features'].permute(0,2,1)

            #    y_output = domain_cls(yt.permute(0,2,1), ts=t_int)
            #    domain_y = y_output['domain_cls']
            #    feat_y = y_output['features'].permute(0,2,1)
            #    print(domain_y.argmax(dim=1), "!!!!!!!!!!!!")
            #    if True:
            #        if args.lambda_s > 0:
            #            domain_loss = F.cross_entropy(domain_y,
            #                    domain_y.new_zeros((len(domain_y),)).long())
            #        else:
            #            domain_loss = 0
            #    else:
            #        domain_loss = F.cosine_similarity(feat_x, feat_y, dim=-1).mean()

            #    # Si (faithful expert)

            #    if args.lambda_i > 0:
            #        if args.voxelization:
            #            eps = phi(yt, t, **model_kwargs)
            #            x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
            #                        furthest_point_idx]
            #            y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
            #            vox_x, _, norm = voxelization(x.permute(0,2,1),
            #                    x.permute(0,2,1))
            #            vox_y = voxelization(y0_est.permute(0,2,1),
            #                    y0_est.permute(0,2,1), norm=norm)[0] # batch_size x 3 x res x res x res
            #            subdist_loss = F.mse_loss(vox_y, vox_x)
            #        elif False:
            #            xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
            #                    furthest_point_idx]
            #            yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx),
            #                    furthest_point_idx]
            #            subdist_loss = (xt_sub - yt_sub).pow(2).sum(-1).mean()
            #            dist1, dist2, *_ = chamfer_dist_fn(xt_sub, yt)
            #            #print(dist1.shape)
            #            #subdist_loss = dist1.mean()
            #        else:
            #            eps = phi(yt, t, **model_kwargs)
            #            x_sub = x[torch.arange(len(x)).view(-1, 1).to(furthest_point_idx),
            #                        furthest_point_idx]
            #            y0_est = 1 / alpha_t * (yt  - sigma_t * eps)
            #            dist1_, dist2_, *_ = chamfer_dist_fn(x_sub, y0_est)
            #            subdist_loss = dist1_.mean()
            #    else:
            #        subdist_loss = 0
            #    grad = torch.autograd.grad(
            #        (args.lambda_s * domain_loss + args.lambda_i * subdist_loss),
            #        yt, allow_unused=True)[0]
            #    #print(grad[grad.nonzero(as_tuple=True)])
            #    #print("!!!!!!!!!!!", len(grad.nonzero(as_tuple=True)[0]),
            #    #"!!!!!!!!")
            #    if args.model == 'pvd':
            #        model.eval()
            #    return grad

            @torch.enable_grad()
            def cond_fn_sub_naive(yt, t, phi, model_kwargs):
                gamma_t = \
                    model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)
                noise = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                xt = x * alpha_t + noise * sigma_t
                xt_sub = xt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]
                yt_sub = yt[torch.arange(len(x)).view(-1,1).to(furthest_point_idx), furthest_point_idx]

                loss = (xt_sub - yt_sub).pow(2).mean()
                grad = torch.autograd.grad(loss, yt, allow_unused=True)[0]
                return grad


            @torch.enable_grad()
            def cond_fn_sub(xt, t, phi, model_kwargs):
                model.dynamics.requires_grad_(True)

                #sub_x = \
                #    pointnet2_utils.gather_operation(x.transpose(1,2).contiguous(), furthest_point_idx).transpose(1, 2).contiguous()

                xt.requires_grad_(True)
                eps = phi(xt, t, **model_kwargs)
                # batch_size x N x 3

                gamma_t = model.inflate_batch_array(model.gamma(t), x)
                alpha_t = model.alpha(gamma_t, x)
                sigma_t = model.sigma(gamma_t, x)

                x0_est = 1 / alpha_t * (xt  - sigma_t * eps)
                sub_x0_est = x0_est[torch.arange(len(x)).view(-1,
                    1).to(furthest_point_idx),
                        furthest_point_idx]
                #sub_x0_est = \
                #    pointnet2_utils.gather_operation(x0_est.transpose(1,2).contiguous(), furthest_point_idx).transpose(1, 2).contiguous()

                loss = args.guidance_scale * (sub_x0_est - sub_x).pow(2).mean()

                grad = torch.autograd.grad(loss, xt, allow_unused=True)[0]
                return grad

            x_ori = x
            z_t_list = []
            x_edit_list = []
            x_edit_list_y = []
            preds_list = []
            preds_list_y = []

            for k in range(K):
                eps = model.sample_combined_position_feature_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )
                z = alpha_t * x + sigma_t * eps # diffusion
                z_t_list.append(z)
                y = z.detach().clone()
                for _ in tqdm(range(int(args.t * args.diffusion_steps))):
                    z, noise = model.sample_p_zs_given_zt(t - 1./args.diffusion_steps, t, z,
                        node_mask, None, None, fix_noise=False,
                        cond_fn=cond_fn_entropy if args.entropy_guided else (cond_fn_egsde if args.egsde else (cond_fn_sub
                        if args.guidance_scale > 0 else None)), #noise=noise,
                        return_noise=True)
                    if args.egsde or args.entropy_guided:
                        y = model.sample_p_zs_given_zt(t - 1./args.diffusion_steps, t, y,
                            node_mask, None, None, fix_noise=False,
                            cond_fn=None, noise=noise) #return_noise=True)
                    t = t - 1. / args.diffusion_steps
                    print('diff', (z-y).abs().max(), (z-y).abs().mean())
                assert torch.all(t.abs() < 1e-5) #torch.allclose(t, torch.zeros_like(t))
                x_edit = model.sample_p_x_given_z0(z, node_mask, None, None,
                        fix_noise=False)
                x = x_edit

                if args.egsde or args.entropy_guided:
                    x_edit_y = model.sample_p_x_given_z0(y, node_mask, None, None,
                            fix_noise=False)
                    y = x_edit_y

                logits = \
                    classifier(scale_to_unit_cube_torch(x_edit.clone().detach()).permute(0, 2, 1), activate_DefRec=False)
                    #classifier(x_edit.clone().detach().permute(0, 2, 1), activate_DefRec=False)

                preds = logits["cls"].max(dim=1)[1]
                preds_val = logits["cls"].softmax(-1).max(dim=1)[0]
                preds_label = [idx_to_label[int(d)] for d in preds]

                print(k, preds_label)
                preds_list.append(zip(preds_label, preds_val))
                #x = x - x.mean(dim=1, keepdim=True)
                #norm = x.pow(2).sum(-1).sqrt().max(-1).values
                #x /= norm[:, None, None]
                x_edit_list.append(x_edit)
                t = args.t * x.new_ones((x.shape[0], 1), device=x.device) # reset

                if args.egsde or args.entropy_guided:
                    logits_y = \
                        classifier(scale_to_unit_cube_torch(x_edit_y.clone().detach()).permute(0, 2, 1), activate_DefRec=False)
                        #classifier(x_edit.clone().detach().permute(0, 2, 1), activate_DefRec=False)
                    preds_y = logits_y["cls"].max(dim=1)[1]
                    preds_val_y = logits_y["cls"].softmax(-1).max(dim=1)[0]
                    preds_label_y = [idx_to_label[int(d)] for d in preds_y]
                    print("EGSDE" if args.egsde else "ENTROPY_GUIDED")
                    print(k, preds_label_y)
                    preds_list_y.append(zip(preds_label_y, preds_val_y))
                    x_edit_list_y.append(x_edit_y)


            logits = classifier(scale_to_unit_cube_torch(x_ori.clone().detach()).permute(0, 2, 1), activate_DefRec=False)
            #logits = classifier(x_ori.clone().detach().permute(0, 2, 1), activate_DefRec=False)
            preds = logits["cls"].max(dim=1)[1]
            preds_val = logits["cls"].softmax(-1).max(dim=1)[0]
            preds_label = [idx_to_label[int(d)] for d in preds]

            #for b, (pred_label, pred_val)  in enumerate(zip(preds_label, preds_val)):

            for iter, (z_t, x_edit, preds_edit) in enumerate(zip(z_t_list, x_edit_list,
                preds_list)):
                rgbs_edit = get_color(x_edit.cpu().numpy())

                if args.egsde or args.entropy_guided:
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

                    if args.egsde or args.entropy_guided:
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
