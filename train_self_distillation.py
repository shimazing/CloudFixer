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
from losses import InfoNCELoss, softmax_entropy
import torch.nn as nn
import torch.nn.functional as F
from utils import random_rotate_one_axis_torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from gnn import GCN

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
parser.add_argument('--n_epochs', type=int, default=50)
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
parser.add_argument('--num_workers', type=int, default=8, help='Number of worker for the dataloader')
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
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
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
parser.add_argument('--mode', nargs='+', type=str, default=['step1'])
parser.add_argument('--dataset_src', type=str, default='shapenet')
parser.add_argument('--dataset_tgt', type=str, default='modelnet')
parser.add_argument('--keep_sub', action='store_true', help='Disable wandb') # TODO
parser.add_argument('--n_subsample', type=int, default=64)
#parser.add_argument('--classifier', type=str,
#    default='../GAST/experiments/GAST_balanced_unitnorm_randomscale0.2/model.ptdgcnn')
parser.add_argument('--self_ensemble', action='store_true')
########################## CL
parser.add_argument('--lambda_cl', default=1, type=float)
parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--em_temperature', default=2.5, type=float)
parser.add_argument('--threshold', default=0.8, type=float)
parser.add_argument('--with_dm', action='store_true',
    help='whether to obtain diffusion model views')
parser.add_argument('--rotation', type=eval, default=True)
parser.add_argument('--src_random_remove', action='store_true', help='remove random part from src')
parser.add_argument('--use_ori', action='store_true', help='remove random part from src')
parser.add_argument('--clf_guidance', action='store_true',)
parser.add_argument('--lambda_clf', default=100.0, type=float) #action='store_true',)
parser.add_argument('--deterministic_val', action='store_true', default=True) # make classifier deterministic
parser.add_argument('--tgt_train_mode', default='pseudo_label',
        choices=['pseudo_label', 'entropy_minimization'])
parser.add_argument('--dm_resume',
    default='outputs/unit_std_pvd_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995/generative_model_ema_last.npy')
##### model
parser.add_argument('--gn', action='store_true', default=False) # make classifier deterministic
parser.add_argument('--bn', type=str, default='bn') # 'bn' or 'ln'
parser.add_argument('--input_transform', action='store_true',
    help='whether to apply input_transform (rotation) in DGCNN')
parser.add_argument('--fc_norm', action='store_true', default=False)
parser.add_argument('--cl', action='store_true', help='whether to use cl head')
parser.add_argument('--cl_dim', default=1024, type=int)
parser.add_argument('--cl_norm', default=False, type=eval)
parser.add_argument('--time_cond', action='store_true', default=True)
parser.add_argument('--interval', type=int, default=5) # for step2
parser.add_argument('--elastic_distortion', type=eval, default=True)
parser.add_argument('--random_scale', type=eval, default=True)
parser.add_argument('--ssl', type=str, default='ori')

args = parser.parse_args()
#if args.input_transform or not args.time_cond:
#    args.gn = True # deterministic
args.random_remove = False
if args.dataset_tgt == 'scannet':
    args.random_remove = args.src_random_remove

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

#args.random_scale = True
#args.elastic_distortion = True
#args.src_random_remove = False
if args.n_nodes == 1024:
    dataset_dict = {'shapenet': ShapeNet, 'scannet': ScanNet, 'modelnet': ModelNet}
    # train
    dataset_src = dataset_dict[args.dataset_src](io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=args.random_scale,
            random_rotation=True, zero_mean=not args.no_zero_mean,
            random_remove=args.random_remove,
            self_distillation=True,
            elastic_distortion=args.elastic_distortion,
            p_keep=0.7,
            )
    dataset_tgt = dataset_dict[args.dataset_tgt](io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=args.random_scale,
            random_rotation=True, zero_mean=not args.no_zero_mean,
            elastic_distortion=args.elastic_distortion,
            self_distillation=True)
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
            drop_last=True, num_workers=args.num_workers, shuffle=True)
    train_loader_tgt = DataLoader(dataset_tgt, batch_size=args.batch_size,
            sampler=None,
            drop_last=True, num_workers=args.num_workers, shuffle=True)
    train_loader_tgt_iter = iter(train_loader_tgt)
    train_loader_tgt_noshuffle = DataLoader(dataset_tgt, batch_size=args.batch_size,
            sampler=None,
            drop_last=False, num_workers=args.num_workers, shuffle=False)

    val_loader_src = DataLoader(dataset_src_val, batch_size=args.batch_size,
            sampler=None,
            drop_last=False, num_workers=args.num_workers)
    val_loader_tgt = DataLoader(dataset_tgt_val, batch_size=args.batch_size,
            sampler=None,
            drop_last=False, num_workers=args.num_workers)


args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

args.exp_name = \
        f'SDist_{args.t}_fc_norm{args.fc_norm}{args.bn}gn{args.gn}_{args.dataset_src}2{args.dataset_tgt}_withDM{args.with_dm}_dm{args.model}_timecond{args.time_cond}_clf_guidance{args.clf_guidance}{args.lambda_clf}_aug{args.random_remove}{args.jitter}{args.elastic_distortion}{args.rotation}{args.random_scale}_useOri{args.use_ori}_epochs{args.n_epochs}_cl{args.cl}{args.cl_dim}{args.cl_norm}_{args.ssl}_step1'
    #f'SDist_{args.t}_fc_norm{args.fc_norm}bn{args.bn}gn{args.gn}_{args.dataset_src}2{args.dataset_tgt}_withDM{args.with_dm}_dm{args.model}_timecond{args.time_cond}_clf_guidance{args.clf_guidance}{args.lambda_clf}_srcRandomRemove{args.random_remove}_useOri{args.use_ori}_epochs{args.n_epochs}_step1'

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
if args.with_dm:
    model.load_state_dict(torch.load(args.dm_resume, map_location='cpu'))
    model = torch.nn.DataParallel(model)
model.eval()
model_ = model.module if args.with_dm else model

ori_model = args.model
args.model = 'dgcnn'
classifier = DGCNN(args).to(device).train()
print(classifier)
classifier_ema = copy.deepcopy(classifier)
ema = flow_utils.EMA(args.ema_decay)

#classifier.module.load_state_dict(torch.load(f"outputs/{args.exp_name}/last.pt"))
#print("Load the last model")

classifier = torch.nn.DataParallel(classifier)
classifier_ema = torch.nn.DataParallel(classifier_ema)
if args.bn == 'bn':
    from sync_batchnorm import convert_model
    classifier = convert_model(classifier).to(device)
    classifier_ema = convert_model(classifier_ema).to(device)
classifier_ema.eval()

class BYOL(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=256, proj_dim=1024):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x, target_x):
        x = F.normalize(self.predictor(x), dim=1)
        target_x = F.normalize(target_x)
        return (2 - 2 * (x*target_x).sum(dim=-1)).mean()


class Distillation_Loss(nn.Module):
    def __init__(self, out_dim=10, teacher_temp=0.1, student_temp=0.5):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        teacher_out = F.softmax((teacher_output/ self.teacher_temp), dim=-1)
        total_loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        return total_loss.mean()

dist_loss = Distillation_Loss() if args.ssl == 'ori' else BYOL().to(device)
optim = get_optim(args, classifier, ssl_loss=dist_loss)
scheduler = CosineAnnealingLR(optim, args.n_epochs)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr,
        epochs=args.n_epochs,
        steps_per_epoch=int(len(train_loader_src)/args.accum_grad))
criterion = nn.CrossEntropyLoss()
infonce = InfoNCELoss(args.temperature)
args.model = ori_model

def sigmoid_rampup(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(current_epoch, weight=1):
    return weight * sigmoid_rampup(current_epoch, args.n_epochs)


@torch.no_grad()
def update_pseudo_labels():
    classifier.eval()
    all_labels = []
    for data in train_loader_tgt_noshuffle:
        batch_size = len(data[0])
        cls_logits = 0
        for i in range(args.K):
            pcs = data[0].to(device).permute(0,2,1)
            node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
            t_int = torch.randint(
                0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
                size=(data[0].size(0), 1), device=device).float()
            if i == 0:
                t_int = torch.zeros_like(t_int)
            t = t_int / args.diffusion_steps
            gamma_t = model_.inflate_batch_array(model_.gamma(t),
                    pcs.permute(0,2,1))
            alpha_t = model_.alpha(gamma_t, pcs.permute(0,2,1))
            sigma_t = model_.sigma(gamma_t, pcs.permute(0,2,1))
            eps = model_.sample_combined_position_feature_noise(
                n_samples=pcs.size(0), n_nodes=pcs.size(2),
                node_mask=node_mask,
                device=device
            ).permute(0,2,1)
            pcs_t = alpha_t * pcs + sigma_t * eps
            logits = classifier(pcs_t, ts=t_int.flatten())
            cls_logits = cls_logits + logits['cls']
        pseudo_label = cls_logits.argmax(dim=1) # soft voting
        all_labels.append(pseudo_label)
    all_labels = torch.cat(all_labels, dim=0)
    assert len(all_labels) == len(dataset_tgt)
    return all_labels

if 'step2' in args.mode:
    gnn = GCN(num_features=1024, num_classes=10).to(device)
    classifier.load_state_dict(torch.load(f"outputs/{args.exp_name}/best.pt"))
    print("Load the best model")
    #args.exp_name = 'SDist' + args.exp_name[16:] + "_step2"
    args.exp_name = \
        f'SDist_{args.t}_fc_norm{args.fc_norm}{args.bn}gn{args.gn}_{args.dataset_src}2{args.dataset_tgt}_withDM{args.with_dm}_dm{args.model}_timecond{args.time_cond}_clf_guidance{args.clf_guidance}{args.lambda_clf}_aug{args.random_remove}{args.jitter}{args.elastic_distortion}{args.rotation}{args.random_scale}_useOri{args.use_ori}_epochs{args.n_epochs}_cl{args.cl}{args.cl_dim}{args.cl_norm}_{args.ssl}_step2'
        #f'SDist_{args.t}_fc_norm{args.fc_norm}{args.bn}gn{args.gn}_{args.dataset_src}2{args.dataset_tgt}_withDM{args.with_dm}_dm{args.model}_timecond{args.time_cond}_clf_guidance{args.clf_guidance}{args.lambda_clf}_aug{args.src_random_remove}{args.jitter}{args.elastic_distortion}{args.rotation}_useOri{args.use_ori}_epochs{args.n_epochs}_cl{args.cl}{args.cl_dim}{args.cl_norm}_{args.ssl}_step2'
        #f'SDist_{args.t}_fc_norm{args.fc_norm}{args.bn}gn{args.gn}_{args.dataset_src}2{args.dataset_tgt}_withDM{args.with_dm}_dm{args.model}_timecond{args.time_cond}_clf_guidance{args.clf_guidance}{args.lambda_clf}_aug{args.src_random_remove}{args.jitter}{args.elastic_distortion}{args.rotation}_useOri{args.use_ori}_epochs{args.n_epochs}_cl{args.cl}{args.cl_dim}{args.cl_norm}_step2'
        #f'SDist_{args.t}_fc_norm{args.fc_norm}{args.bn}gn{args.gn}_{args.dataset_src}2{args.dataset_tgt}_withDM{args.with_dm}_dm{args.model}_timecond{args.time_cond}_clf_guidance{args.clf_guidance}{args.lambda_clf}_aug{args.src_random_remove}{args.jitter}{args.elastic_distortion}{args.rotation}_useOri{args.use_ori}_epochs{args.n_epochs}_step2'
    if not os.path.exists(f'outputs/{args.exp_name}'):
        os.makedirs(f'outputs/{args.exp_name}')
    confident = torch.zeros(len(dataset_tgt)).bool().to(device) #torch.zeros((0)).to(device)
    tgt_pseudo_labels = update_pseudo_labels()
    # reinitialize classifier
    ori_model = args.model
    args.model = 'dgcnn'
    classifier = DGCNN(args).to(device).train()
    print(classifier)
    classifier_ema = copy.deepcopy(classifier)
    ema = flow_utils.EMA(args.ema_decay)
    #classifier.module.load_state_dict(torch.load(f"outputs/{args.exp_name}/last.pt"))
    #print("Load the last model")
    classifier = torch.nn.DataParallel(classifier)
    classifier_ema = torch.nn.DataParallel(classifier_ema)
    if args.bn == 'bn':
        from sync_batchnorm import convert_model
        classifier = convert_model(classifier).to(device)
        classifier_ema = convert_model(classifier_ema).to(device)
    classifier_ema.eval()
    optim = get_optim(args, classifier)
    scheduler = CosineAnnealingLR(optim, args.n_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr,
            epochs=args.n_epochs,
            steps_per_epoch=int(len(train_loader_src)/args.accum_grad))
    criterion = nn.CrossEntropyLoss()
    criterion_noreduction = nn.CrossEntropyLoss(reduction='none')
    infonce = InfoNCELoss(args.temperature)
    args.model = ori_model

def reinitialize(gnn):
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
    gnn.apply(weights_init)

def refine_labels(epoch):
    global gnn
    global confident
    global tgt_pseudo_labels
    classifier.eval()
    probs_lst = []
    feats_lst = []
    preds_lst = []
    with torch.no_grad():
        for data in tqdm(train_loader_tgt_noshuffle):
            pcs = data[0].permute(0,2,1).to(device)
            node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
            ind = data[4]
            cls_probs = 0
            feats = 0
            cls_preds = []
            for k in range(args.K):
                t_int = torch.randint(
                    0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
                    size=(pcs.size(0), 1), device=device).float()
                if k == 0:
                    t_int = torch.zeros_like(t_int)
                t = t_int / args.diffusion_steps
                gamma_t = model_.inflate_batch_array(model_.gamma(t),
                        pcs.permute(0,2,1))
                alpha_t = model_.alpha(gamma_t, pcs.permute(0,2,1))
                sigma_t = model_.sigma(gamma_t, pcs.permute(0,2,1))
                eps = model_.sample_combined_position_feature_noise(
                    n_samples=pcs.size(0), n_nodes=pcs.size(2),
                    node_mask=node_mask,
                    device=device
                ).permute(0,2,1)
                pcs_t = alpha_t * pcs + sigma_t * eps
                logits = classifier(pcs_t, ts=t_int.flatten())
                cls_logits = logits['cls']
                cls_probs = cls_probs + F.softmax(cls_logits, dim=1)
                cls_preds.append(cls_logits.argmax(dim=1))
                feats = feats + logits['global_features']
            cls_probs = cls_probs / args.K
            feats = feats / args.K
            cls_preds = torch.stack(cls_preds, dim=1)
            cls_preds = torch.mode(cls_preds, dim=1).values
            probs_lst.append(cls_probs)
            feats_lst.append(feats)
            preds_lst.append(cls_preds)
    probs = torch.cat(probs_lst, dim=0)
    feats = torch.cat(feats_lst, dim=0)
    preds = torch.cat(preds_lst, dim=0)
    assert len(feats) == len(dataset_tgt)
    # construct graph
    tau = 0.95
    clip = 10

    corr_matrix = np.corrcoef(feats.cpu())
    corr_matrix_bin = corr_matrix.copy()

    corr_matrix = torch.from_numpy(corr_matrix).to(device)
    corr_matrix_bin[corr_matrix_bin<=tau] = 0
    corr_matrix_bin[corr_matrix_bin>tau] = 1
    corr_matrix_bin = torch.from_numpy(corr_matrix_bin).bool().to(device)

    print("Construct Graph")
    edges = []
    for i in range(len(feats)):
        indexes = torch.argsort(corr_matrix[i], descending=True)
        tau_mask = corr_matrix_bin[i]==1
        tau_mask = tau_mask[indexes]
        indexes = indexes[tau_mask][:clip]
        for index in set(indexes):
            edges.append((i, index))
    edge_index = torch.tensor(edges).transpose(1, 0).long().to(device)
    # construct graph done
    reinitialize(gnn)
    optimizer_gnn = torch.optim.Adam(gnn.parameters(), lr=0.001, weight_decay=5e-4)
    gnn.train()
    print("Train GNN")
    for iter in tqdm(range(1000)):
        optimizer_gnn.zero_grad()  # Clear gradients.
        column_vec = torch.rand(feats.shape[0]).reshape(-1, 1) < 0.2
        column_vec = column_vec.repeat(1, 10)
        pl_features = probs.clone()
        pl_features[column_vec] = 0  # randomly drop probability feature

        _, out = gnn(feats, pl_features, edge_index) # Perform a singùle forward pass.
        loss_gnn = criterion(out, preds.long())  # Compute the loss solely based on the training nodes.
        loss_gnn.backward()
        optimizer_gnn.step()  # Update parameters based on
    # reinit label using gnn
    print("Update labels and confident set")
    gnn.eval()
    p = 1 - epoch / args.n_epochs
    with torch.no_grad():
        _, out = gnn(feats, probs, edge_index)
        out = F.softmax(out, dim=1)
        scores, preds = out.max(dim=1)  # U
        selected = torch.ones_like(scores).bool()
        #new_confident_gnn = self.filter_data(self.predictions_gnn.cpu(), self.scores_gnn.cpu(), 1-self.current_epoch/100, 10)
        # filter data
        for cls in range(10): # n_classes
            scores_cls = scores[preds == cls]
            if len(scores_cls) == 0:
                continue
            thrs_ind = min(int(round(len(scores_cls) * p)), len(scores_cls) - 1)
            sorted_score, _ = torch.sort(scores_cls)
            thres = sorted_score[thrs_ind]
            selected[(preds == cls) & (scores < thres)] = False
        confident = confident | selected # 원래 confident 였거나 새로 confident 해졌거나
    tgt_pseudo_labels[confident] = preds[confident]


import time

def main():
    best_val = 0
    best_val_tgt = 0
    best_tgt = 0
    threshold = args.threshold
    optim.zero_grad()
    for epoch in range(args.n_epochs):
        src_correct = 0
        src_est_correct = 0
        tgt_correct = 0
        tgt_est_correct = 0
        src_count = 0
        tgt_count = 0
        # Train
        #train_count = 0
        #train_correct = 0
        #cl_correct = 0
        #cl_count = 0
        n_selected = 0
        selected_count = 0
        #if epoch>0:
        #    for param in classifier.module.named_parameters():
        #        print(param)
        #        break
        #    input()
        start = time.time()
        for i, data_src in enumerate(train_loader_src):
            #print("loading src", time.time() - start)
            #start = time.time()
            classifier.train()
            try:
                data_tgt = next(train_loader_tgt_iter)
            except:
                train_loader_tgt_iter = iter(train_loader_tgt)
                data_tgt = next(train_loader_tgt_iter)
            #print("loading tgt", time.time() - start)
            #start = time.time()
            @torch.enable_grad()
            def cond_fn_clf_guidance(yt, t_int, labels):
                # with dm
                yt.requires_grad_(True)
                logits = classifier(yt, ts=t_int.flatten())
                cls_logits = logits['cls']
                cls_loss = criterion(cls_logits, labels)
                grad = torch.autograd.grad(args.lambda_clf * cls_loss, yt)[0]
                return grad

            label_src = data_src[1].to(device)
            #print("label_src", label_src)
            label_tgt_gt = data_tgt[1].to(device)
            if 'step2' in args.mode:
                tgt_idx = data_tgt[4]
                label_tgt_pseudo = tgt_pseudo_labels[tgt_idx]
                tgt_weight = confident[tgt_idx].float() + 0.2 * (1-confident[tgt_idx].float())
                tgt_weight = tgt_weight.to(device)
            #weight = torch.cat((torch.ones_like(label_src).float(),
            #                    tgt_weight), dim=0)
            #print("label_tgt_gt", label_tgt_gt)
            n_src = len(data_src[0])
            n_tgt = len(data_tgt[0])
            src_count += n_src
            tgt_count += n_tgt
            data = torch.cat((
                data_src[0].to(device), #.permute(0, 2, 1),
                data_tgt[0].to(device) #.permute(0, 2, 1)
                ),dim=0)
            data_aug = torch.cat((
                data_src[3].to(device),
                data_tgt[3].to(device)
                ), dim=0)
            pcs = data.permute(0,2,1) # weak aug
            pcs_aug = data_aug.permute(0,2,1) # strong aug

            node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
            #t_int = torch.randint(
            #    0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
            #    size=(pcs.size(0), 1), device=device).float()

            t_int = torch.randint(
                0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
                size=(pcs.size(0), 1), device=device).float()
            if not args.use_ori:
                t = t_int / args.diffusion_steps
                gamma_t = model_.inflate_batch_array(model_.gamma(t),
                        pcs.permute(0,2,1))
                alpha_t = model_.alpha(gamma_t, pcs.permute(0,2,1))
                sigma_t = model_.sigma(gamma_t, pcs.permute(0,2,1))
                eps = model_.sample_combined_position_feature_noise(
                    n_samples=pcs.size(0), n_nodes=pcs.size(2),
                    node_mask=node_mask,
                    device=device
                ).permute(0,2,1)
                pcs_t = alpha_t * pcs + sigma_t * eps
            else:
                pcs_t = pcs
                t_int = torch.zeros_like(t_int)
                # TODO: DGCNN에 t conditioning 하기
            with torch.no_grad():
                classifier.eval()
                logits = classifier(
                        pcs_t.contiguous(), ts=t_int.flatten()
                    ) #, activate_DefRec=False)
                cls_logits = logits['cls']
                src_cls_loss_weak = criterion(cls_logits[:n_src],
                        label_src)
                # src_correct
                src_correct += (cls_logits[:n_src].argmax(dim=1) ==
                        label_src).float().sum().item()
                tgt_correct += (cls_logits[n_src:].argmax(dim=1) ==
                        label_tgt_gt).float().sum().item()
                cls_logits_weak = cls_logits
                classifier.train()

            with torch.no_grad():
                logits_ema = classifier_ema(
                    pcs_t.contiguous(), ts=t_int.flatten()
                    ) #, activate_DefRec=False)


            t_int2 = torch.randint(
                0, max(int(args.diffusion_steps * args.t),1), # ~= diffusion_steps * 0.4
                size=(pcs.size(0), 1), device=device).float()
            t2 = t_int2 / args.diffusion_steps
            gamma_t2 = model_.inflate_batch_array(model_.gamma(t2),
                    pcs.permute(0,2,1))
            alpha_t2 = model_.alpha(gamma_t2, pcs.permute(0,2,1))
            sigma_t2 = model_.sigma(gamma_t2, pcs.permute(0,2,1))
            eps2 = model_.sample_combined_position_feature_noise(
                n_samples=pcs.size(0), n_nodes=pcs.size(2),
                node_mask=node_mask,
                device=device
            ).permute(0,2,1)

            pcs_t2 = alpha_t2 * pcs_aug + sigma_t2 * eps2
            # TODO: DGCNN에 t conditioning 하기
            logits2 = classifier(
                pcs_t2.contiguous(), ts=t_int2.flatten()
                ) #, activate_DefRec=False)

            cls_logits = logits_ema['cls']
            cls_logits2 = logits2['cls']
            if args.ssl == 'ori':
                cl_feat = logits_ema['global_features']
                cl_feat2 = logits2['global_features']
            else: #'byol'
                cl_feat = logits_ema['cl_feat']
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
                    eps1_ = model(x=pcs_t.permute(0,2,1), node_mask=node_mask,
                            t=t,
                            phi=True).permute(0,2,1)
                    if args.clf_guidance:
                        grad = cond_fn_clf_guidance(pcs_t,
                                t_int,
                                torch.cat((label_src, label_tgt), dim=0))
                        # src
                        eps1_[:n_src] = eps1_[:n_src] + sigma_t[:n_src] * grad[:n_src]
                        # tgt
                        tgt_selected_idx = selected_final.nonzero(as_tuple=True)[0]+n_src
                        eps1_[tgt_selected_idx] = eps1_[tgt_selected_idx] +\
                                sigma_t[tgt_selected_idx] * grad[tgt_selected_idx]
                    pcs_est = (pcs_t - sigma_t * eps1_) / alpha_t
                    #if args.rotation:
                    pcs_est = \
                        random_rotate_one_axis_torch(pcs_est.permute(0,2,1),
                                axis='z').permute(0,2,1)

                logits_est = classifier(
                    pcs_est.contiguous(), ts=torch.zeros_like(t_int).flatten()
                    )
                cls_logits_est = logits_est['cls']
                cl_feat_est = logits_est['global_features']
            # src loss
            src_cls_loss = \
                    criterion(cls_logits2[:n_src], label_src) + \
                    (criterion(cls_logits_est[:n_src], label_src) if args.with_dm
                    else 0)

            # tgt loss
            if 'step1' in args.mode:
                pass
                if args.tgt_train_mode == 'pseudo_label':
                    if torch.any(selected_final):
                        tgt_cls_loss = criterion(cls_logits_weak[n_src:][selected_final],
                                label_tgt[selected_final])# + \
                              #0.2 * criterion(cls_logits_weak[n_src:][~selected_final],
                              #  label_tgt[~selected_final]) #+ \
                                #criterion(cls_logits2[n_src:][selected_final],
                                #    label_tgt2[selected_final])
                        if args.with_dm:
                            selected_est = max_prob_est > threshold
                            selected_est_final = selected & selected_est & (label_tgt == label_tgt_est)
                            tgt_cls_loss = tgt_cls_loss + \
                                criterion(cls_logits_est[n_src:][selected_est_final],
                                    label_tgt_est[selected_est_final])
                    else:
                        tgt_cls_loss = 0
                else: # entropy_minimization
                    tgt_cls_loss = \
                        softmax_entropy(cls_logits_weak[n_src:]/args.em_temperature) + \
                        softmax_entropy(cls_logits2[n_src:]/args.em_temperature) + \
                        (softmax_entropy(cls_logits_est[n_src:]/args.em_temperature)
                                if args.with_dm else 0)
            elif 'step2' in args.mode:
                tgt_cls_loss = (criterion_noreduction(cls_logits2[n_src:],
                    label_tgt_pseudo) * tgt_weight).mean()


            # ssl
            ssl_loss = dist_loss(cl_feat2, cl_feat) + \
                    (dist_loss(cl_feat_est, cl_feat) if args.with_dm else 0)


            total_loss = ssl_loss * get_current_consistency_weight(epoch) + \
                src_cls_loss #+ tgt_cls_loss #+ src_cls_loss_weak + tgt_cls_loss
            if 'step2' in args.mode:
                total_loss = total_loss + tgt_cls_loss
            (total_loss/args.accum_grad).backward()
            if (i+1) % args.accum_grad  == 0:
                optim.step()
                optim.zero_grad()
                optim_step = True
                ema.update_model_average(classifier_ema, classifier)
                if True: #'step2' in args.mode:
                    scheduler.step()
            else:
                optim_step = False
            #for param in classifier.module.named_parameters():
            #    print(param)
            #    break

            if args.with_dm:
                src_est_correct += (cls_logits_est[:n_src].argmax(dim=1) ==
                        label_src).float().sum().item()
            if args.with_dm:
                tgt_est_correct += (cls_logits_est[n_src:].argmax(dim=1) ==
                        label_tgt_gt).float().sum().item()

            if (i+1) % 10 == 0:
                #print(f'Epoch {epoch} {i} src acc {src_correct / src_count} tgt acc {tgt_correct / tgt_count}')
                print(f'Epoch {epoch} {i} src acc {src_correct / src_count} tgt acc {tgt_correct / tgt_count} selected {n_selected / selected_count}')
                print(total_loss.item(), ssl_loss.item() *
                        get_current_consistency_weight(epoch),
                        src_cls_loss.item(), tgt_cls_loss)
                print(get_current_consistency_weight(epoch))
            #print(time.time() - start, "for loop")
            #start = time.time()
        if not optim_step:
            optim.step()
            optim.zero_grad()
            ema.update_model_average(classifier_ema, classifier)
            if True: #'step2' in args.mode:
                scheduler.step()
        if False: #'step1' in args.mode:
            scheduler.step()
        #print(f'Epoch {epoch} {i} src acc {src_correct / src_count} tgt acc {tgt_correct / tgt_count}')
        print(f'Epoch {epoch} {i} src acc {src_correct / src_count} tgt acc {tgt_correct / tgt_count} selected {n_selected / selected_count}')
        print(total_loss.item(), ssl_loss.item() *
                get_current_consistency_weight(epoch),
                src_cls_loss.item(), tgt_cls_loss)

        if 'step2' in args.mode and (epoch + 1) % args.interval == 0:
            print("Refine Labels")
            refine_labels(epoch)


        # Val : accuracy check
        with torch.no_grad():
            if (epoch + 1) % 1 == 0:
                classifier.eval()
                n_correct_src = 0
                n_correct_tgt = 0
                n_total_src = 0
                n_total_tgt = 0
                for val_src in val_loader_src:
                    pcs = val_src[0].to(device).permute(0,2,1)
                    t_int = torch.randint(
                        0, max(int(args.diffusion_steps * args.t) if not
                            args.deterministic_val else 1, 1), # ~= diffusion_steps * 0.4
                        size=(pcs.size(0), 1), device=device).float()
                    t = t_int / args.diffusion_steps

                    gamma_t = model_.inflate_batch_array(model_.gamma(t),
                            pcs.permute(0,2,1))
                    alpha_t = model_.alpha(gamma_t, pcs.permute(0,2,1))
                    sigma_t = model_.sigma(gamma_t, pcs.permute(0,2,1))

                    node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
                    eps = model_.sample_combined_position_feature_noise(
                        n_samples=pcs.size(0), n_nodes=pcs.size(2),
                        node_mask=node_mask,
                        device=device
                    ).permute(0,2,1)

                    pcs_t = alpha_t * pcs + sigma_t * eps if not \
                        args.deterministic_val else pcs
                    logits_src = classifier(pcs_t, ts=t_int.flatten())
                    #print(logits_src['domain_cls'].argmax(dim=1), "source")
                    n_correct_src += (logits_src['cls'].argmax(dim=1) ==
                            val_src[1].to(device)).float().sum().item()
                    n_total_src += len(logits_src['cls'])

                for val_tgt in val_loader_tgt:
                    pcs = val_tgt[0].to(device).permute(0,2,1)
                    t_int = torch.randint(
                        0, max(int(args.diffusion_steps * args.t) if not
                            args.deterministic_val else 1,1), # ~= diffusion_steps * 0.4
                        size=(pcs.size(0), 1), device=device).float()
                    t = t_int / args.diffusion_steps

                    gamma_t = model_.inflate_batch_array(model_.gamma(t),
                            pcs.permute(0,2,1))
                    alpha_t = model_.alpha(gamma_t, pcs.permute(0,2,1))
                    sigma_t = model_.sigma(gamma_t, pcs.permute(0,2,1))

                    node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
                    eps = model_.sample_combined_position_feature_noise(
                        n_samples=pcs.size(0), n_nodes=pcs.size(2),
                        node_mask=node_mask,
                        device=device
                    ).permute(0,2,1)

                    pcs_t = alpha_t * pcs + sigma_t * eps if not \
                            args.deterministic_val else pcs
                    logits_tgt = classifier(pcs_t, ts=t_int.flatten())
                    #print(logits_tgt['domain_cls'].argmax(dim=1), "target")
                    n_correct_tgt += (logits_tgt['cls'].argmax(dim=1) ==
                            val_tgt[1].to(device)).float().sum().item()
                    n_total_tgt += len(logits_tgt['cls'])
                print("epoch", epoch)
                print("source acc", n_correct_src / n_total_src)
                print('target acc', n_correct_tgt / n_total_tgt)

                val_result = n_correct_src / n_total_src #+ n_correct_tgt / n_total_tgt
                tgt_result = n_correct_tgt / n_total_tgt
                if tgt_result > best_tgt:
                    best_tgt = tgt_result
                print("best_tgt", best_tgt)

                if val_result > best_val:
                    best_val = val_result
                    best_val_tgt = tgt_result
                    torch.save(classifier.state_dict(),
                            f'outputs/{args.exp_name}/best.pt')
                    torch.save(optim.state_dict(),
                            f'outputs/{args.exp_name}/best_optim.pt')
                print("best_val", best_val, "best_val_tgt", best_val_tgt)
                if (epoch + 1) % 10 == 0:
                    torch.save(classifier.state_dict(),
                        f'outputs/{args.exp_name}/{epoch+1}.pt')
                    torch.save(optim.state_dict(),
                        f'outputs/{args.exp_name}/{epoch+1}_optim.pt')
                    torch.save(classifier_ema.state_dict(),
                        f'outputs/{args.exp_name}/{epoch+1}_ema.pt')
                torch.save(classifier.state_dict(),
                        f'outputs/{args.exp_name}/last.pt')
                torch.save(classifier_ema.state_dict(),
                        f'outputs/{args.exp_name}/last_ema.pt')
                torch.save(optim.state_dict(),
                        f'outputs/{args.exp_name}/last_optim.pt')
            print(args.exp_name)
            classifier.train()

    classifier.load_state_dict(torch.load(f"outputs/{args.exp_name}/best.pt"))
    print("Load the best model")

    with torch.no_grad():
        classifier.eval()
        n_correct_src = 0
        n_correct_tgt = 0
        n_total_src = 0
        n_total_tgt = 0
        for val_src in val_loader_src:
            pcs = val_src[0].to(device).permute(0,2,1)
            t_int = torch.randint(
                0, max(int(args.diffusion_steps * args.t) if not
                    args.deterministic_val else 1, 1), # ~= diffusion_steps * 0.4
                size=(pcs.size(0), 1), device=device).float()
            t = t_int / args.diffusion_steps

            gamma_t = model_.inflate_batch_array(model_.gamma(t),
                    pcs.permute(0,2,1))
            alpha_t = model_.alpha(gamma_t, pcs.permute(0,2,1))
            sigma_t = model_.sigma(gamma_t, pcs.permute(0,2,1))

            node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
            eps = model_.sample_combined_position_feature_noise(
                n_samples=pcs.size(0), n_nodes=pcs.size(2),
                node_mask=node_mask,
                device=device
            ).permute(0,2,1)

            pcs_t = alpha_t * pcs + sigma_t * eps if not \
                args.deterministic_val else pcs
            logits_src = classifier(pcs_t, ts=t_int.flatten())
            #print(logits_src['domain_cls'].argmax(dim=1), "source")
            n_correct_src += (logits_src['cls'].argmax(dim=1) ==
                    val_src[1].to(device)).float().sum().item()
            n_total_src += len(logits_src['cls'])

        for val_tgt in val_loader_tgt:
            pcs = val_tgt[0].to(device).permute(0,2,1)
            t_int = torch.randint(
                0, max(int(args.diffusion_steps * args.t) if not
                    args.deterministic_val else 1,1), # ~= diffusion_steps * 0.4
                size=(pcs.size(0), 1), device=device).float()
            t = t_int / args.diffusion_steps

            gamma_t = model_.inflate_batch_array(model_.gamma(t),
                    pcs.permute(0,2,1))
            alpha_t = model_.alpha(gamma_t, pcs.permute(0,2,1))
            sigma_t = model_.sigma(gamma_t, pcs.permute(0,2,1))

            node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
            eps = model_.sample_combined_position_feature_noise(
                n_samples=pcs.size(0), n_nodes=pcs.size(2),
                node_mask=node_mask,
                device=device
            ).permute(0,2,1)

            pcs_t = alpha_t * pcs + sigma_t * eps if not \
                    args.deterministic_val else pcs
            logits_tgt = classifier(pcs_t, ts=t_int.flatten())
            #print(logits_tgt['domain_cls'].argmax(dim=1), "target")
            n_correct_tgt += (logits_tgt['cls'].argmax(dim=1) ==
                    val_tgt[1].to(device)).float().sum().item()
            n_total_tgt += len(logits_tgt['cls'])
        #print("epoch", epoch)
        print("source acc (best)", n_correct_src / n_total_src)
        print('target acc (best)', n_correct_tgt / n_total_tgt)


if __name__ == "__main__":
    main()
