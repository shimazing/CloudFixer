# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import pandas as pd
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
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
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine, linear')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
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
parser.add_argument('--scale', type=float, default=1)
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
parser.add_argument('--time_cond', action='store_true', default=True)

args = parser.parse_args()

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
    # train
    dataset_src = ShapeNet(io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=True)
    dataset_tgt = ScanNet(io, './data', 'train', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=True)
    # val
    dataset_src_val = ShapeNet(io, './data', 'val', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=False)
    dataset_tgt_val = ScanNet(io, './data', 'val', jitter=args.jitter,
            scale=args.scale,
            scale_mode=args.scale_mode,
            random_scale=False,
            random_rotation=False)

    #if args.dataset == 'scannet':
    #    test_dataset = ScanNet(io, './data', 'test', jitter=args.jitter,
    #        scale=args.scale,
    #        scale_mode=args.scale_mode,
    #        random_rotation=False) # for classification
    #elif args.dataset == 'modelnet':
    #    test_dataset = ModelNet(io, './data', 'test', jitter=args.jitter,
    #        scale=args.scale,
    #        scale_mode=args.scale_mode,
    #        random_rotation=False) # for classification
    #elif args.dataset == 'shapenet':
    #    test_dataset = ShapeNet(io, './data', 'test', jitter=args.jitter,
    #        scale=args.scale,
    #        scale_mode=args.scale_mode,
    #        random_rotation=False) # for classification

    # TODO jitter??!!
    #train_dataset_sampler, val_dataset_sampler = split_set(dataset_,
    #    domain='shapenet')

    train_loader_src = DataLoader(dataset_src, batch_size=args.batch_size,
            sampler=None, #train_dataset_sampler,
            drop_last=True, num_workers=16)
    train_loader_tgt = DataLoader(dataset_tgt, batch_size=args.batch_size,
            sampler=None, #train_dataset_sampler,
            drop_last=True, num_workers=16)
    train_loader_tgt_iter = iter(train_loader_tgt)

    val_loader_src = DataLoader(dataset_src_val, batch_size=args.batch_size,
            sampler=None, #train_dataset_sampler,
            drop_last=False)
    val_loader_tgt = DataLoader(dataset_tgt_val, batch_size=args.batch_size,
            sampler=None, #train_dataset_sampler,
            drop_last=False)

    #val_loader = DataLoader(dataset_, batch_size=args.batch_size,
    #        sampler=val_dataset_sampler) #, drop_last=True)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
    #        shuffle=False, drop_last=False)
    #test_loader_vis = DataLoader(test_dataset, batch_size=args.batch_size,
    #        shuffle=False, drop_last=False,
    #        sampler=ImbalancedDatasetSampler(test_dataset))

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

# alpha, sigma 계산을 위해
time_cond = args.time_cond
if time_cond:
    dataset_info = None
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, None) #dataloaders['train'])
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    model = model.to(device)

args.model = 'dgcnn'
classifier = DGCNN(args).to(device).eval()
classifier = torch.nn.DataParallel(classifier)
optim = get_optim(args, classifier)


def main():
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val = 0
    for epoch in range(args.n_epochs):
        # Train
        train_count = 0
        train_correct = 0
        for i, data_src in enumerate(tqdm(train_loader_src)):
            classifier.train()
            try:
                data_tgt = next(train_loader_tgt_iter)
            except:
                train_loader_tgt_iter = iter(train_loader_tgt)
                data_tgt = next(train_loader_tgt_iter)
            data = torch.stack((
                data_src[0].to(device).permute(0, 2, 1),
                data_tgt[0].to(device).permute(0, 2, 1)),dim=1)
            data = data.view(-1, data.shape[-2], data.shape[-1])
            if time_cond:
                pcs = data

                t_int = torch.randint(
                    0, int(model.T), # ~= diffusion_steps * 0.4
                    size=(pcs.size(0), 1), device=device).float()
                t = t_int / model.T

                gamma_t = model.inflate_batch_array(model.gamma(t),
                        pcs.permute(0,2,1))
                alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
                sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

                eps = model.sample_combined_position_feature_noise(
                    n_samples=pcs.size(0), n_nodes=pcs.size(2), node_mask=None,
                    device=device
                ).permute(0,2,1)

                pcs_t = alpha_t * pcs + sigma_t * eps
                # TODO: DGCNN에 t conditioning 하기
                logits = classifier(
                    pcs_t, ts=t_int.flatten()
                    ) #, activate_DefRec=False)
            else:
                # bn 때문에 한번에 pass 해주도록 수정함
                logits = classifier(
                    data
                    ) #, activate_DefRec=False)
            logits = logits['domain_cls']
            labels = torch.stack((logits.new_zeros((len(logits)//2,)),
                logits.new_ones((len(logits)//2,))),dim=1).long()
            labels = labels.flatten()
            train_correct += (logits.argmax(dim=1) == labels).float().sum().item()
            train_count += len(logits)

            loss = loss_fn(logits, labels)
            print(loss.item())
            loss.backward()
            optim.step()
            optim.zero_grad()
            print(f"Epoch {epoch} Train acc {train_correct/train_count}")

            # Val : accuracy check
        with torch.no_grad():
            if (epoch + 1) % 1 == 0:
                classifier.eval()
                n_correct_src = 0
                n_correct_tgt = 0
                n_total_src = 0
                n_total_tgt = 0
                for val_src in val_loader_src:
                    if time_cond:
                        pcs = val_src[0].to(device).permute(0,2,1)

                        t_int = torch.randint(
                            0, model.T, # ~= diffusion_steps * 0.4
                            size=(pcs.size(0), 1), device=device).float()
                        t = t_int / model.T

                        gamma_t = model.inflate_batch_array(model.gamma(t),
                                pcs.permute(0,2,1))
                        alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
                        sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

                        eps = model.sample_combined_position_feature_noise(
                            n_samples=pcs.size(0), n_nodes=pcs.size(2), node_mask=None,
                            device=device
                        ).permute(0,2,1)

                        pcs_t = alpha_t * pcs + sigma_t * eps
                        logits_src = classifier(pcs_t, ts=t_int.flatten())
                    else:
                        logits_src = classifier(val_src[0].to(device).permute(0,2,1))
                    #print(logits_src['domain_cls'].argmax(dim=1), "source")
                    n_correct_src += (logits_src['domain_cls'].argmax(dim=1) ==
                            0).float().sum().item()
                    n_total_src += len(logits_src['domain_cls'])

                for val_tgt in val_loader_tgt:
                    if time_cond:
                        pcs = val_tgt[0].to(device).permute(0,2,1)

                        t_int = torch.randint(
                            0, 200, # ~= diffusion_steps * 0.4
                            size=(pcs.size(0), 1), device=device).float()
                        t = t_int / model.T

                        gamma_t = model.inflate_batch_array(model.gamma(t),
                                pcs.permute(0,2,1))
                        alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
                        sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

                        eps = model.sample_combined_position_feature_noise(
                            n_samples=pcs.size(0), n_nodes=pcs.size(2), node_mask=None,
                            device=device
                        ).permute(0,2,1)

                        pcs_t = alpha_t * pcs + sigma_t * eps
                        logits_tgt = classifier(pcs_t, ts=t_int.flatten())
                    else:
                        logits_tgt = classifier(val_tgt[0].to(device).permute(0,2,1))
                    #print(logits_tgt['domain_cls'].argmax(dim=1), "target")
                    n_correct_tgt += (logits_tgt['domain_cls'].argmax(dim=1) ==
                            1).float().sum().item()
                    n_total_tgt += len(logits_tgt['domain_cls'])
                print("epoch", epoch)
                print("source acc", n_correct_src / n_total_src)
                print('target acc', n_correct_tgt / n_total_tgt)

                val_result = n_correct_src / n_total_src + n_correct_tgt / n_total_tgt

                if val_result > best_val:
                    best_val = val_result
                    torch.save(classifier.module.state_dict(),
                            "outputs/domain_classifier_DGCNN_shape_scan_timecondGN_fullt.pt")

if __name__ == "__main__":
    main()
