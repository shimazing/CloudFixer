import pandas as pd
import copy
import utils
import argparse
import wandb
import os
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
from losses import InfoNCELoss
import torch.nn as nn
import torch.nn.functional as F
from utils import random_rotate_one_axis_torch

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
parser.add_argument('--model', type=str, default='pvd')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')
# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine, linear')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--no_zero_mean', action='store_true', default=True)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_nodes', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-3)
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
## EGNN args -->
#parser.add_argument('--n_layers', type=int, default=6,
#                    help='number of layers')
#parser.add_argument('--inv_sublayers', type=int, default=1,
#                    help='number of layers')
#parser.add_argument('--nf', type=int, default=256, #128,
#                    help='number of layers')
#parser.add_argument('--tanh', type=eval, default=True,
#                    help='use tanh in the coord_mlp')
#parser.add_argument('--attention', type=eval, default=True,
#                    help='use attention in the EGNN')
#parser.add_argument('--norm_constant', type=float, default=1,
#                    help='diff/(|diff| + norm_constant)')
#parser.add_argument('--sin_embedding', type=eval, default=False,
#                    help='whether using or not the sin embedding')
## <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
#parser.add_argument('--filter_n_atoms', type=int, default=None,
#                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
#parser.add_argument('--dequantization', type=str, default='argmax_variational',
#                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str, default='mazing')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb',
        default=True) # TODO
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
#parser.add_argument("--conditioning", nargs='+', default=[],
#                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
#parser.add_argument('--resume', type=str,
#        default="outputs/pointnet/generative_model_last.npy", #required=True,
#                    #help='outputs/unit_val_shapenet_pointnet_resume/generative_model_last.npy'
#                    )
#parser.add_argument('--dynamics_config', type=str,
#        default='pointnet2/exp_configs/mvp_configs/config_standard_ori.json')
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
parser.add_argument('--scale', type=float, default=1)
########################
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--output_pts', type=int, default=512)
######################## guided sampling
#parser.add_argument('--guidance_scale', type=float, default=0)
parser.add_argument('--mode', nargs='+', type=str, default=['eval'])
parser.add_argument('--dataset_src', type=str, default='shapenet')
parser.add_argument('--dataset_tgt', type=str, default='scannet')
parser.add_argument('--keep_sub', action='store_true', help='Disable wandb') # TODO
parser.add_argument('--n_subsample', type=int, default=64)
#parser.add_argument('--classifier', type=str,
#    default='../GAST/experiments/GAST_balanced_unitnorm_randomscale0.2/model.ptdgcnn')
parser.add_argument('--self_ensemble', action='store_true')
parser.add_argument('--time_cond', action='store_true') #, default=True)
parser.add_argument('--fc_norm', action='store_true', default=False)
########################## CL
parser.add_argument('--cl', action='store_true', help='whether to use cl head')
parser.add_argument('--cl_dim', default=1024, type=int)
parser.add_argument('--lambda_cl', default=1, type=float)
parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--input_transform', action='store_true',
    help='whether to apply input_transform (rotation) in DGCNN')
parser.add_argument('--with_dm', action='store_true',
    help='whether to obtain diffusion model views')
parser.add_argument('--rotation', action='store_true',)
parser.add_argument('--clf_guidance', action='store_true',)
parser.add_argument('--lambda_clf', action='store_true',)
parser.add_argument('--gn', action='store_true',) # make classifier deterministic
parser.add_argument('--dm_resume',
    default='outputs/unit_std_pvd_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995/generative_model_ema_last.npy')

args = parser.parse_args()
if args.input_transform or not args.time_cond:
    args.gn = True # deterministic

io = log.IOStream(args)

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
    dataset_dict = {'shapenet': ShapeNet, 'scannet': ScanNet, 'modelnet': ModelNet}
    # train
    dataset_src = dataset_dict[args.dataset_src](io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=True, zero_mean=not args.no_zero_mean)
    dataset_tgt = dataset_dict[args.dataset_tgt](io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=True, zero_mean=not args.no_zero_mean)
    # val
    dataset_src_val = dataset_dict[args.dataset_src](io, './data', 'val', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=False, zero_mean=not args.no_zero_mean)
    dataset_tgt_val = dataset_dict[args.dataset_tgt](io, './data', 'test', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=False, zero_mean=not args.no_zero_mean) # 참고용

    train_loader_src = DataLoader(dataset_src, batch_size=args.batch_size,
            sampler=None,
            drop_last=True, num_workers=args.num_workers)
    train_loader_tgt = DataLoader(dataset_tgt, batch_size=args.batch_size,
            sampler=None,
            drop_last=True, num_workers=args.num_workers)
    train_loader_tgt_iter = iter(train_loader_tgt)

    val_loader_src = DataLoader(dataset_src_val, batch_size=args.batch_size,
            sampler=None,
            drop_last=False, num_workers=args.num_workers)
    val_loader_tgt = DataLoader(dataset_tgt_val, batch_size=args.batch_size,
            sampler=None,
            drop_last=False, num_workers=args.num_workers)


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

args.exp_name = \
    f'latent_classifier_{args.t}_fc_norm{args.fc_norm}_{args.dataset_src}2{args.dataset_tgt}_withDM{args.with_dm}_dm{args.model}_cl{args.cl}{args.cl_dim}lam{args.lambda_cl}_temperature{args.temperature}_inputTrans{args.input_transform}_rotationAug{args.rotation}_timecond{args.time_cond}_gn{args.gn}'

if not os.path.exists(f'outputs/{args.exp_name}'):
    os.makedirs(f'outputs/{args.exp_name}')

# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name , 'project':
        'latent_clf', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

time_cond = args.time_cond
model = get_model(args, device)
model = model.to(device)
model.load_state_dict(torch.load(args.dm_resume, map_location='cpu'))
model = torch.nn.DataParallel(model)
model.eval()

ori_model = args.model
args.model = 'dgcnn'
classifier = DGCNN(args).to(device).eval()
classifier = torch.nn.DataParallel(classifier)
optim = get_optim(args, classifier)
criterion = nn.CrossEntropyLoss()
infonce = InfoNCELoss(args.temperature)
args.model = ori_model

def main():
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val = 0
    threshold = 0.8
    for epoch in range(args.n_epochs):
        # Train
        train_count = 0
        train_correct = 0
        src_correct = 0
        src_est_correct = 0
        tgt_correct = 0
        tgt_est_correct = 0
        src_count = 0
        tgt_count = 0
        cl_correct = 0
        cl_count = 0

        n_selected = 0
        selected_count = 0
        for i, data_src in enumerate(train_loader_src):
            classifier.train()
            try:
                data_tgt = next(train_loader_tgt_iter)
            except:
                train_loader_tgt_iter = iter(train_loader_tgt)
                data_tgt = next(train_loader_tgt_iter)

            @torch.enable_grad()
            def cond_fn_clf_guidance(yt, t_int, labels):
                # with dm
                yt.requires_grad_(True)
                logits = classifier(yt, ts=t_int.flatten())
                cls_logits = logits['cls']
                cls_loss = criterion(cls_logits, labels)
                grad = torch.autograd.grad(args.lambda_clf * cls_loss, yt)[0]
                return grad

            #data = torch.stack((
            #    data_src[0].to(device).permute(0, 2, 1),
            #    data_tgt[0].to(device).permute(0, 2, 1)),dim=1)
            #data = data.view(-1, data.shape[-2], data.shape[-1])
            label_src = data_src[1].to(device)
            label_tgt_gt = data_tgt[1].to(device)
            n_src = len(data_src[0])
            n_tgt = len(data_tgt[0])
            src_count += n_src
            tgt_count += n_tgt
            data = torch.cat((
                data_src[0].to(device), #.permute(0, 2, 1),
                data_tgt[0].to(device) #.permute(0, 2, 1)
                ),dim=0)
            if True: #time_cond:
                pcs = data.permute(0,2,1) #,2)
                t_int = torch.randint(
                    0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
                    size=(pcs.size(0), 1), device=device).float()
                t = t_int / args.diffusion_steps
                gamma_t = model.module.inflate_batch_array(model.module.gamma(t),
                        pcs.permute(0,2,1))
                alpha_t = model.module.alpha(gamma_t, pcs.permute(0,2,1))
                sigma_t = model.module.sigma(gamma_t, pcs.permute(0,2,1))
                node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
                eps = model.module.sample_combined_position_feature_noise(
                    n_samples=pcs.size(0), n_nodes=pcs.size(2),
                    node_mask=node_mask,
                    device=device
                ).permute(0,2,1)

                pcs_t = alpha_t * pcs + sigma_t * eps
                # TODO: DGCNN에 t conditioning 하기
                logits = classifier(
                    pcs_t, ts=t_int.flatten()
                    ) #, activate_DefRec=False)

                t_int2 = torch.randint(
                    0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
                    size=(pcs.size(0), 1), device=device).float()
                t2 = t_int2 / args.diffusion_steps
                gamma_t2 = model.module.inflate_batch_array(model.module.gamma(t2),
                        pcs.permute(0,2,1))
                alpha_t2 = model.module.alpha(gamma_t2, pcs.permute(0,2,1))
                sigma_t2 = model.module.sigma(gamma_t2, pcs.permute(0,2,1))
                eps2 = model.module.sample_combined_position_feature_noise(
                    n_samples=pcs.size(0), n_nodes=pcs.size(2),
                    node_mask=node_mask,
                    device=device
                ).permute(0,2,1)

                pcs_t2 = alpha_t2 * pcs + sigma_t2 * eps2
                if args.rotation:
                    pcs_t2 = random_rotate_one_axis_torch(pcs_t2.permute(0,2,1),
                            axis='z').permute(0,2,1)
                # TODO: DGCNN에 t conditioning 하기
                logits2 = classifier(
                    pcs_t2, ts=t_int2.flatten()
                    ) #, activate_DefRec=False)
                cls_logits = logits['cls']
                cls_logits2 = logits2['cls']
                cl_feat = logits['cl_feat']
                cl_feat2 = logits2['cl_feat']
                max_prob, label_tgt = F.softmax(cls_logits[n_src:], dim=-1).max(dim=-1)
                max_prob2, label_tgt2 = F.softmax(cls_logits2[n_src:], dim=-1).max(dim=-1)
                selected = max_prob > threshold
                selected2 = max_prob2 > threshold
                selected_final = selected & selected2 & (label_tgt == label_tgt2)
                n_selected += selected_final.float().sum().item()
                selected_count += len(selected_final)
                if args.with_dm:
                    with torch.no_grad():
                        #node_mask = pcs_t2.new_ones(pcs_t2.shape[0],
                        #        pcs_t2.shape[2]).unsqueeze(-1)
                        eps2_ = model(x=pcs_t2.permute(0,2,1), node_mask=node_mask, t=t2,
                                phi=True).permute(0,2,1)
                        if args.clf_guidance:
                            grad = cond_fn_clf_guidance(pcs_t2,
                                    t_int2,
                                    torch.cat((label_src, label_tgt), dim=0))
                            # src
                            eps2_[:n_src] = eps2_[:n_src] + sigma_t2[:n_src] * grad[:n_src]
                            # tgt
                            tgt_selected_idx = selected_final.nonzero(as_tuple=True)[0]+n_src
                            eps2_[tgt_selected_idx] = eps2_[tgt_selected_idx] +\
                                    sigma_t2[tgt_selected_idx] * grad[tgt_selected_idx]
                        pcs_est = (pcs_t2 - sigma_t2 * eps2_) / alpha_t2
                        if args.rotation:
                            pcs_est = \
                                random_rotate_one_axis_torch(pcs_est.permute(0,2,1),
                                        axis='z').permute(0,2,1)

                    logits_est = classifier(
                        pcs_est, ts=torch.zeros_like(t_int).flatten()
                        )
                #if args.with_dm:
                    cls_logits_est = logits_est['cls']
            else:
                # bn 때문에 한번에 pass 해주도록 수정함
                logits = classifier(
                    data.permute(0,2,1)
                    ) #, activate_DefRec=False)
            if args.with_dm:
                cl_feat_est = logits_est['cl_feat']

            # src loss
            src_cls_loss = criterion(cls_logits[:n_src], label_src) + \
                    criterion(cls_logits2[:n_src], label_src) + \
                    (criterion(cls_logits_est[:n_src], label_src) if args.with_dm
                    else 0)
            # Pseudo Labeling (SPST)
            #max_prob, label_tgt = F.softmax(cls_logits[n_src:], dim=-1).max(dim=-1)
            #max_prob2, label_tgt2 = F.softmax(cls_logits2[n_src:], dim=-1).max(dim=-1)
            if args.with_dm:
                max_prob_est, label_tgt_est = F.softmax(cls_logits_est[n_src:], dim=-1).max(dim=-1)


            if torch.any(selected_final):
                tgt_cls_loss = criterion(cls_logits[n_src:][selected_final],
                        label_tgt[selected_final]) + \
                        criterion(cls_logits2[n_src:][selected_final],
                            label_tgt2[selected_final])
                if args.with_dm:
                    selected_est = max_prob_est > threshold
                    selected_est_final = selected & selected_est & (label_tgt == label_tgt_est)
                    tgt_cls_loss = tgt_cls_loss + \
                        criterion(cls_logits_est[n_src:][selected_est_final],
                            label_tgt_est[selected_est_final])
            else:
                tgt_cls_loss = 0

            # constrastive learning (infoNCE loss)
            if args.with_dm:
                cl_loss = (infonce(torch.stack((cl_feat, cl_feat2), dim=1)) \
                    + infonce(torch.stack((cl_feat, cl_feat_est), dim=1)) \
                    + infonce(torch.stack((cl_feat2, cl_feat_est), dim=1)))/3
            else:
                cl_loss = infonce(torch.stack((cl_feat, cl_feat2), dim=1))

            total_loss = args.lambda_cl * cl_loss + src_cls_loss + tgt_cls_loss
            (total_loss/args.accum_grad).backward()
            if (i+1) % args.accum_grad  == 0:
                optim.step()
                optim.zero_grad()
                optim_step = True
            else:
                optim_step = False

            # src_correct
            src_correct += (cls_logits[:n_src].argmax(dim=1) ==
                    label_src).float().sum().item()
            if args.with_dm:
                src_est_correct += (cls_logits_est[:n_src].argmax(dim=1) ==
                        label_src).float().sum().item()
            # tgt_correct <- 참고용으로만 ,,ㅎㅎ
            tgt_correct += (cls_logits[n_src:].argmax(dim=1) ==
                    label_tgt_gt).float().sum().item()
            if args.with_dm:
                tgt_est_correct += (cls_logits_est[n_src:].argmax(dim=1) ==
                        label_tgt_gt).float().sum().item()
            # cl acc
            cl_correct += ((cl_feat.unsqueeze(1) *
                    cl_feat2.unsqueeze(0)).sum(-1).argmax(dim=-1) == \
                torch.arange(len(cl_feat)).to(cl_feat.device)).float().sum().item()
            cl_count += len(cl_feat)

            if (i+1) % 10 == 0:
                print(f'Epoch {epoch} {i} src acc {src_correct / src_count} tgt acc {tgt_correct / tgt_count} cl acc {cl_correct / cl_count} selected {n_selected / selected_count}')
        if not optim_step:
            optim.step()
            optim.zero_grad()
        print(f'Epoch {epoch} {i} src acc {src_correct / src_count} tgt acc {tgt_correct / tgt_count} cl acc {cl_correct / cl_count} selected {n_selected / selected_count}' )


        # Val : accuracy check
        with torch.no_grad():
            if (epoch + 1) % 1 == 0:
                classifier.eval()
                n_correct_src = 0
                n_correct_tgt = 0
                n_total_src = 0
                n_total_tgt = 0
                for val_src in val_loader_src:
                    if True: #time_cond:
                        pcs = val_src[0].to(device).permute(0,2,1)
                        t_int = torch.randint(
                            0, max(int(args.diffusion_steps * args.t), 1), # ~= diffusion_steps * 0.4
                            size=(pcs.size(0), 1), device=device).float()
                        t = t_int / args.diffusion_steps

                        gamma_t = model.module.inflate_batch_array(model.module.gamma(t),
                                pcs.permute(0,2,1))
                        alpha_t = model.module.alpha(gamma_t, pcs.permute(0,2,1))
                        sigma_t = model.module.sigma(gamma_t, pcs.permute(0,2,1))

                        node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
                        eps = model.module.sample_combined_position_feature_noise(
                            n_samples=pcs.size(0), n_nodes=pcs.size(2),
                            node_mask=node_mask,
                            device=device
                        ).permute(0,2,1)

                        pcs_t = alpha_t * pcs + sigma_t * eps
                        logits_src = classifier(pcs_t, ts=t_int.flatten())
                    else:
                        logits_src = classifier(val_src[0].to(device).permute(0,2,1))
                    #print(logits_src['domain_cls'].argmax(dim=1), "source")
                    n_correct_src += (logits_src['cls'].argmax(dim=1) ==
                            val_src[1].to(device)).float().sum().item()
                    n_total_src += len(logits_src['cls'])

                for val_tgt in val_loader_tgt:
                    if True: #time_cond:
                        pcs = val_tgt[0].to(device).permute(0,2,1)
                        t_int = torch.randint(
                            0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
                            size=(pcs.size(0), 1), device=device).float()
                        t = t_int / args.diffusion_steps

                        gamma_t = model.module.inflate_batch_array(model.module.gamma(t),
                                pcs.permute(0,2,1))
                        alpha_t = model.module.alpha(gamma_t, pcs.permute(0,2,1))
                        sigma_t = model.module.sigma(gamma_t, pcs.permute(0,2,1))

                        node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
                        eps = model.module.sample_combined_position_feature_noise(
                            n_samples=pcs.size(0), n_nodes=pcs.size(2),
                            node_mask=node_mask,
                            device=device
                        ).permute(0,2,1)

                        pcs_t = alpha_t * pcs + sigma_t * eps
                        logits_tgt = classifier(pcs_t, ts=t_int.flatten())
                    else:
                        logits_tgt = classifier(val_tgt[0].to(device).permute(0,2,1))
                    #print(logits_tgt['domain_cls'].argmax(dim=1), "target")
                    n_correct_tgt += (logits_tgt['cls'].argmax(dim=1) ==
                            val_tgt[1].to(device)).float().sum().item()
                    n_total_tgt += len(logits_tgt['cls'])
                print("epoch", epoch)
                print("source acc", n_correct_src / n_total_src)
                print('target acc', n_correct_tgt / n_total_tgt)

                val_result = n_correct_src / n_total_src #+ n_correct_tgt / n_total_tgt

                if val_result > best_val:
                    best_val = val_result
                    torch.save(classifier.module.state_dict(),
                            f'outputs/{args.exp_name}/best.pt')
                    torch.save(optim.state_dict(),
                            f'outputs/{args.exp_name}/best_optim.pt')
                if (epoch + 1) % 10 == 0:
                    torch.save(classifier.module.state_dict(),
                        f'outputs/{args.exp_name}/{epoch+1}.pt')
                    torch.save(optim.state_dict(),
                        f'outputs/{args.exp_name}/{epoch+1}_optim.pt')
                torch.save(classifier.module.state_dict(),
                        f'outputs/{args.exp_name}/last.pt')
                torch.save(optim.state_dict(),
                        f'outputs/{args.exp_name}/last_optim.pt')


if __name__ == "__main__":
    main()
