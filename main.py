import copy
import utils
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
from data.dataloader_Norm import ShapeNetCore, ModelNet40C
from utils_GAST import pc_utils_Norm, log
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from sampling import sample
from tqdm import tqdm
import pandas as pd
import random

parser = argparse.ArgumentParser(description='Diffusion for PC')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--dataset', type=str, default='shapenet')
parser.add_argument('--model', type=str, default='pointnet',
        choices=['pointnet', 'pvd', 'transformer'])
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
parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_nodes', type=int, default=1024)
parser.add_argument('--lr', type=float, default=2e-4)
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
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str, default='mazing')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
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
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.9999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--random_scale', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--jitter', type=eval, default=True)
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--out_path', type=str, default='./exps')
parser.add_argument('--knn', type=int, default=32)
parser.add_argument('--accum_grad', type=int, default=1)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--scale_mode', type=str, default='unit_val',
        choices=['unit_val', 'unit_std', 'unit_norm'])
parser.add_argument('--dynamics_config', type=str,
        default='pointnet2/exp_configs/mvp_configs/config_standard_attention_real_3072_partial_points_rot_90_scale_1.2_translation_0.1.json',
        )
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--test_ema', action='store_true')
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--no_zero_mean', action='store_true')
parser.add_argument('--lr_gamma', default=1, type=float)
parser.add_argument('--cls_uniform', default=True, type=eval)
args = parser.parse_args()

if args.dataset.startswith('modelnet40'):
    args.n_cls = 40
else:
    args.n_cls = 10

zero_mean = not args.no_zero_mean

io = log.IOStream(args)


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
    if args.mode == 'train':
        if args.dataset == 'shapenet':
            dataset_ = ShapeNet(io, './data', 'train', jitter=args.jitter,
                    scale=args.scale, scale_mode=args.scale_mode,
                    random_scale=args.random_scale, zero_mean=zero_mean)
            dataset_val = ShapeNet(io, './data', 'val', jitter=args.jitter,
                    scale=args.scale, scale_mode=args.scale_mode,
                    zero_mean=zero_mean)
        elif args.dataset == 'modelnet':
            dataset_ = ModelNet(io, './data', 'train', jitter=args.jitter,
                    scale=args.scale, scale_mode=args.scale_mode,
                    random_scale=args.random_scale, zero_mean=zero_mean)
            dataset_val = ModelNet(io, './data', 'val', jitter=args.jitter,
                    scale=args.scale, scale_mode=args.scale_mode,
                    zero_mean=zero_mean)
        elif args.dataset == 'scannet':
            dataset_ = ScanNet(io, './data', 'train', jitter=args.jitter,
                    scale=args.scale, scale_mode=args.scale_mode,
                    random_scale=args.random_scale, zero_mean=zero_mean)
            dataset_val = ScanNet(io, './data', 'val', jitter=args.jitter,
                    scale=args.scale, scale_mode=args.scale_mode,
                    zero_mean=zero_mean)
        elif args.dataset.startswith('modelnet40'):
            dataset_ = ModelNet40C(split='train', corruption='original',
                    num_classes=args.n_cls, random_scale=args.random_scale)
            dataset_val = ModelNet40C(split='val', corruption='original',
                    num_classes=args.n_cls, random_scale=False)
        train_dataset_sampler, val_dataset_sampler = None, None #split_set(dataset_)
        train_loader = DataLoader(dataset_, batch_size=args.batch_size,
                sampler=None if not args.cls_uniform else ImbalancedDatasetSampler(dataset_), #train_dataset_sampler,
                drop_last=True,
                shuffle=True if not args.cls_uniform else False,
                num_workers=args.num_workers)
        val_loader = DataLoader(dataset_val, batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                sampler=val_dataset_sampler,
                num_workers=args.num_workers)
    else:
        dataset_test = ShapeNet(io, './data', 'test', jitter=False, #args.jitter,
                scale=args.scale, scale_mode=args.scale_mode,
                random_rotation=False, random_scale=False, zero_mean=zero_mean)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False)

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
        split='test',
        scale_mode='shape_bbox', #args.scale_mode,
    )
    print("ShapeNetCore")

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=2*args.batch_size,
            shuffle=False, drop_last=False)


args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

def normalize_point_clouds(pcs, mode):
    print("normalize", mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    try:
        with open(join(args.resume, 'args.pickle'), 'rb') as f:
            args_ = pickle.load(f)
    except:
        pass



    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')
# Create Model
model = get_model(args, device)
model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def main():
    if args.lr_gamma < 1:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, args.lr_gamma)
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model_last.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim_last.npy'))
        if args.lr_gamma < 1:
            scheduler_state_dict = torch.load(join(args.resume, 'lr_scheduler_last.npy'))
            lr_scheduler.load_state_dict(scheduler_state_dict)
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        print("Resume")

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)
        if args.resume is not None:
            ema_state_dict = torch.load(join(args.resume, 'generative_model_ema_last.npy'))
            model_ema.load_state_dict(ema_state_dict)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


    if args.resume is not None and args.mode == 'test':
        from eval_metrics import compute_all_metrics, jsd_between_point_cloud_sets
        with torch.no_grad():
            model.eval()
            gen_pcs = []
            ref_pcs = []
            for data in tqdm(test_loader):
                x_val = data[0]
                x_gen = sample(args, device, model_ema if args.test_ema else model,
                        nodesxsample=torch.zeros(x_val.shape[0]).fill_(args.n_nodes).long(),
                        fix_noise=False)
                gen_pcs.append(x_gen.cpu())
                ref_pcs.append(x_val.cpu())
            gen_pcs = normalize_point_clouds(torch.cat(gen_pcs, dim=0), 'shape_bbox')
            ref_pcs = normalize_point_clouds(torch.cat(ref_pcs, dim=0), 'shape_bbox')
            results = compute_all_metrics(gen_pcs, ref_pcs, batch_size=128)
            results = {k:v.item() for k, v in results.items()}
            jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
            results['jsd'] = jsd
            model.train()
            print(results)
        return

    # TRAIN
    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.lr_gamma < 1:
            print("LR: ", lr_scheduler.get_last_lr())
        start_epoch = time.time()
        train_epoch(args=args, loader=train_loader,
                    epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype,
                    gradnorm_queue=gradnorm_queue, optim=optim)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        if args.lr_gamma < 1:
            if args.dataset == 'modelnet':
                if (epoch + 1) % (args.n_epochs // 10000) == 0:
                    lr_scheduler.step()
            else:
                lr_scheduler.step()

        if hasattr(model.dynamics, 'report_neighbor_stats'):
            pass
            model.dynamics.report_neighbor_stats()

        if epoch % args.test_epochs == 0 and epoch > 0:
            if isinstance(model, en_diffusion.DiffusionModel):
                wandb.log(model.log_info(), commit=True)
            nll_val = test(args=args, loader=val_loader,
                    epoch=epoch, eval_model=model_ema_dp,
                    partition='Val', device=device, dtype=dtype)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                if args.save_model:
                    args.current_epoch = epoch + 1
                    if args.lr_gamma < 1:
                        utils.save_model(lr_scheduler,
                                'outputs/%s/lr_scheduler.npy' % args.exp_name)
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)

            if args.save_model:
                if args.lr_gamma < 1:
                    utils.save_model(lr_scheduler, 'outputs/%s/lr_scheduler_%s.npy' %
                            (args.exp_name, "last"))
                utils.save_model(optim, 'outputs/%s/optim_%s.npy' %
                        (args.exp_name, "last"))
                utils.save_model(model, 'outputs/%s/generative_model_%s.npy' %
                        (args.exp_name, "last"))
                if args.ema_decay > 0:
                    utils.save_model(model_ema,
                            'outputs/%s/generative_model_ema_%s.npy' %
                            (args.exp_name, "last"))
                with open('outputs/%s/args_%s.pickle' % (args.exp_name, "last"), 'wb') as f:
                    pickle.dump(args, f)

            print('Val loss: %.4f' % nll_val)
            print('Best val loss: %.4f' % best_nll_val)
            wandb.log({"Val loss ": nll_val}, commit=True)

            # Generate Point Clouds
            if args.n_nodes == 2048:
              with torch.no_grad():
                model.eval()

                gen_pcs = []
                ref_pcs = []
                for data in tqdm(val_loader):
                    x_val = data[0]
                    x_gen = sample(args, device, model,
                            nodesxsample=torch.zeros(x_val.shape[0]).fill_(args.n_nodes).long(),
                            fix_noise=False)
                    x_gen = (x_gen - x_gen.mean(1, keepdim=True)) / x_gen.flatten(1).std(1)[:, None, None]
                    gen_pcs.append(x_gen)
                    ref_pcs.append(x_val)
                gen_pcs = torch.cat(gen_pcs, dim=0)
                ref_pcs = torch.cat(ref_pcs, dim=0)

                results = compute_all_metrics(gen_pcs.to(device), ref_pcs.to(device),
                        batch_size=32)
                results = {k:v.item() for k, v in results.items()}
                jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
                results['jsd'] = jsd
                model.train()
                print(results)


if __name__ == "__main__":
    main()
