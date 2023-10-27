from os.path import join
import copy
import random
import argparse
import wandb
import time
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataloader import ModelNet40C, PointDA10, GraspNet10, ImbalancedDatasetSampler
from diffusion import diffusion
from diffusion import utils as flow_utils
from train_test import train_epoch, test
from build_model import get_optim, get_model
import utils


def main():
    parser = argparse.ArgumentParser(description='Diffusion for PC')
    parser.add_argument('--exp_name', type=str, default='debug_10')
    parser.add_argument('--dataset', type=str, default='modelnet40')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--model', type=str, default='transformer',
            choices=['transformer'])
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
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_nodes', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dp', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--clip_grad', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--n_report_steps', type=int, default=1)
    parser.add_argument('--wandb_usr', type=str, default='unknown')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save_model', type=eval, default=True,
                        help='save model')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    parser.add_argument('--test_epochs', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None,
                        help='')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='Amount of EMA decay, 0 means off. A reasonable value'
                            ' is 0.999.')
    parser.add_argument('--random_scale', action='store_true')
    parser.add_argument('--include_charges', type=eval, default=True,
                        help='include atom charge or not')
    parser.add_argument('--jitter', type=eval, default=False)
    parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                        help="Can be used to visualize multiple times per epoch")
    parser.add_argument('--out_path', type=str, default='./exps')
    parser.add_argument('--accum_grad', type=int, default=1)
    parser.add_argument('--scale_mode', type=str, default='unit_std',
            choices=['unit_val', 'unit_std', 'unit_norm'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--test_ema', action='store_true')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--no_zero_mean', action='store_true')
    parser.add_argument('--lr_gamma', default=1, type=float)
    parser.add_argument('--cls_uniform', default=True, type=eval)
    args = parser.parse_args()
    args.zero_mean = not args.no_zero_mean

    if args.dataset.startswith('modelnet40'):
        dataset_ = ModelNet40C(args, partition='train')
        dataset_val = ModelNet40C(args, partition='val')
    elif args.dataset in ['modelnet', 'shapenet', 'scannet']:
        dataset_ = PointDA10(args=args, partition='train')
        dataset_val = PointDA10(args=args, partition='val')
    elif args.dataset in ['synthetic', 'kinect', 'realsense']:
        dataset_ = GraspNet10(args=args, partition='train')
        dataset_val = GraspNet10(args=args, partition='val')
    else:
        raise ValueError('UNDEFINED DATASET')
    train_loader = DataLoader(dataset_, batch_size=args.batch_size,
            sampler=None if not args.cls_uniform else ImbalancedDatasetSampler(dataset_),
            drop_last=True,
            shuffle=True if not args.cls_uniform else False,
            num_workers=args.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32

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
        args.exp_name = exp_name
        args.start_epoch = start_epoch
        args.wandb_usr = wandb_usr
        print(args)

    utils.create_folders(args)

    # Wandb config
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'pc_diffusionTTA', 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    # Create Model
    model = get_model(args, device)
    model = model.to(device)
    optim = get_optim(args, model)

    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

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
    torch.cuda.manual_seed_all(random_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    best_nll_val = float('inf')
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.lr_gamma < 1:
            print("LR: ", lr_scheduler.get_last_lr())
        start_time = time.time()
        train_epoch(args=args, loader=train_loader,
                    epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype,
                    gradnorm_queue=gradnorm_queue, optim=optim)
        print(f"Epoch took {time.time() - start_time:.1f} seconds.")
        if args.lr_gamma < 1:
            lr_scheduler.step()

        if hasattr(model.dynamics, 'report_neighbor_stats'):
            pass
            model.dynamics.report_neighbor_stats()

        if epoch % args.test_epochs == 0 and epoch > 0:
            if isinstance(model, diffusion.DiffusionModel):
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



if __name__ == "__main__":
    main()