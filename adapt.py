import argparse
from copy import deepcopy
from tqdm import tqdm
import wandb
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from data.dataloader import *
from diffusion.build_model import get_model
from classifier import models
from utils import logging
from utils.utils import *
from utils.pc_utils import *
from utils.tta_utils import *
from utils.visualizer import visualize_pclist
from utils.chamfer_distance.chamfer_distance import ChamferDistance


def parse_arguments():
    parser = argparse.ArgumentParser(description='CloudFixer')
    parser.add_argument('--mode', nargs='+', type=str, required=True)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

    # experiments
    parser.add_argument('--out_path', type=str, default='./exps')
    parser.add_argument('--exp_name', type=str, default='adaptation')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    parser.add_argument('--wandb_usr', type=str, default='unknown')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')

    # dataset
    parser.add_argument('--dataset', type=str, default='modelnet40c_background_5')
    parser.add_argument('--dataset_dir', type=str, default='../datasets/modelnet40_c/')
    parser.add_argument('--adv_attack', type=eval, default=False)
    parser.add_argument('--scenario', type=str, default='normal')
    parser.add_argument('--imb_ratio', type=float, default=0)

    # classifier
    parser.add_argument('--classifier', type=str, default='DGCNN')
    parser.add_argument('--classifier_dir', type=str, default='outputs/dgcnn_modelnet40_best_test.pth')
    parser.add_argument('--cls_scale_mode', type=str, default='unit_norm')

    # method and hyperparameters
    parser.add_argument('--method', nargs='+', type=str, default=['pre_trans'])
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--batch_size', type=int, default=4)

    # tta hyperparameters
    parser.add_argument('--episodic', type=eval, default=True)
    parser.add_argument('--test_optim', type=str, default='AdamW')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--test_lr', type=float, default=1e-4)
    parser.add_argument('--params_to_adapt', nargs='+', type=str, default=['all'])
    parser.add_argument('--lame_affinity', type=str, required=False, default='rbf') # for LAME
    parser.add_argument('--lame_knn', type=int, required=False, default=5) # for LAME
    parser.add_argument('--lame_max_steps', type=int, required=False, default=1) # for LAME
    parser.add_argument('--sar_ent_threshold', type=float, default=0.4) # for SAR
    parser.add_argument('--sar_eps_threshold', type=float, default=0.05) # for SAR
    parser.add_argument('--memo_num_augs', type=int, required=False, default=4) # for MEMO
    parser.add_argument('--memo_bn_momentum', type=eval, default=1/17) # for memo, dua, ...
    parser.add_argument('--dua_mom_pre', type=float, default=0.1)
    parser.add_argument('--dua_decay_factor', type=float, default=0.94)
    parser.add_argument('--dua_min_mom', type=float, default=0.005)
    parser.add_argument('--bn_stats_prior', type=float, default=0)
    parser.add_argument('--shot_pl_loss_weight', type=float, default=0.3)
    parser.add_argument('--dda_steps', type=int, default=100)
    parser.add_argument('--dda_guidance_weight', type=float, default=6)
    parser.add_argument('--dda_lpf_method', type=str, default='fps')
    parser.add_argument('--dda_lpf_scale', type=float, default=4)

    parser.add_argument('--ours_steps', type=int, default=100)
    parser.add_argument('--ours_guidance_weight', type=float, default=6)
    parser.add_argument('--ours_lpf_method', type=str, default="")
    parser.add_argument('--ours_lpf_scale', type=float, default=4)

    # diffusion model
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')
    parser.add_argument('--diffusion_dir', type=str, default='outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy')

    # diffusion model hyperparameters
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine, linear')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
    parser.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')
    parser.add_argument('--scale_mode', type=str, default='unit_std')
    parser.add_argument('--n_nodes', type=int, default=1024)
    parser.add_argument('--dp', type=eval, default=True, help='True | False')
    parser.add_argument('--knn', type=int, default=20)
    parser.add_argument('--accum_grad', type=int, default=1)
    parser.add_argument('--t', type=float, default=0.4)
    parser.add_argument('--weighted_reg', type=eval, default=False)
    parser.add_argument('--n_update', default=400, type=int)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adamax')
    parser.add_argument('--optim_end_factor', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lam_l', type=float, default=0)
    parser.add_argument('--lam_h', type=float, default=0)
    parser.add_argument('--t_min', type=float, default=0.02)
    parser.add_argument('--t_max', type=float, default=0.2)
    parser.add_argument('--n_iters_per_update', type=int, default=1)
    parser.add_argument('--subsample', type=int, default=1024)
    parser.add_argument('--pow', type=int, default=1)
    parser.add_argument('--denoising_thrs', type=int, default=100)
    parser.add_argument('--save_itmd', type=int, default=0)
    args = parser.parse_args()
    if 'eval' in args.mode:
        args.no_wandb = True
        args.save_itmd = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


@torch.enable_grad()
def pre_trans(args, model, x, mask, ind, verbose=True):
    def knn(x, k=args.knn, mask=None, return_dist=False):
        # mask : [B, N]
        # x : [B, C=3, N]
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  #거리 가장 가까운거 골라야하니까 음수 붙여줌
        # B x N x N

        if mask is not None:
            B_ind, N_ind = (~mask).nonzero(as_tuple=True)
            pairwise_distance[B_ind, N_ind] = -np.inf
            pairwise_distance[B_ind, :, N_ind] = -np.inf
        # (batch_size, num_points, k)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        if return_dist:
            B = x.shape[0]
            N = x.shape[2]
            dist =  -pairwise_distance[torch.arange(B)[:, None, None],
                                torch.arange(N)[None, :, None],
                    idx] + 1e-8
            is_valid = mask[torch.arange(B)[:, None, None], idx]
            dist[~is_valid] = 0
            n_valid = is_valid.float().sum(dim=-1)
            return idx, (dist.sum(dim=-1) / (n_valid-1).clamp(min=1)).detach().clone()
        return idx

    def matching_loss(model, x, step, t=None):
        if t is None:
            t = (args.t_min * min(1, step/args.denoising_thrs) + (1 - min(1, step/args.denoising_thrs)) * max(args.t_min, args.t_max - 0.2)) + 0.2 * torch.rand(x.shape[0], 1).to(x.device)

        if isinstance(model, nn.DataParallel):
            model = model.module

        gamma_t = model.inflate_batch_array(model.gamma(t), x)
        alpha_t = model.alpha(gamma_t, x)
        sigma_t = model.sigma(gamma_t, x)

        node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
        eps = model.sample_noise(n_samples=x.size(0),
            n_nodes=x.size(1),
            node_mask=node_mask,
        )
        z_t = x*alpha_t + eps*sigma_t
        pred_noise = model(z_t, t=t, node_mask=node_mask, phi=True)
        loss = (pred_noise - eps).pow(2).mean()
        return loss, z_t.detach().clone().cpu()

    lr = args.lr
    steps = args.n_update

    delta = nn.Parameter(torch.zeros_like(x))
    rotation = x.new_zeros((x.size(0), 6))
    rotation[:, 0] = 1
    rotation[:, 4] = 1
    rotation = nn.Parameter(rotation)

    optim = torch.optim.Adamax([
        {
            'params': [delta],
            'lr':lr,
        },
        {
            'params': [rotation],
            'lr': 0.02,
        }],
        lr=lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1,
            end_factor=args.optim_end_factor, total_iters=steps)
    _, knn_dist_square_mean = knn(x.transpose(2,1), k=args.knn,
            mask=(mask.squeeze(-1).bool()), return_dist=True)
    knn_dist_square_mean = knn_dist_square_mean[torch.arange(x.size(0))[:,
        None], ind]
    weight = 1/knn_dist_square_mean.pow(args.pow)
    if not args.weighted_reg:
        weight = torch.ones_like(weight)
    weight = weight / weight.sum(dim=-1, keepdim=True) # normalize
    if args.save_itmd:
        itmd = []
        itmd_zt = []
    for step in tqdm(range(steps), desc='pre trans', ncols=100):
        rot = compute_rotation_matrix_from_ortho6d(rotation)
        y = x + delta
        if (step+1) == args.denoising_thrs: # completion stage
            weight = weight * mask.squeeze(-1)
            weight = weight / weight.sum(dim=-1, keepdim=True) # normalize
        y = y - y.mean(dim=1, keepdim=True)
        if (step+1) >= args.denoising_thrs:
            y = y @ rot
        L21_norm = (torch.norm(delta, 2, dim=-1) * weight).sum(dim=1).mean() # B x N
        matching, zt = matching_loss(model, y, step+1)
        if args.save_itmd > 0 and step % args.szave_itmd == 0:
            itmd.append(y.clone().detach().cpu())
            itmd_zt.append(zt.clone().detach().cpu())
        loss = matching + ((args.lam_h * np.cos(step/steps*np.pi/2) +
            args.lam_l * (1-np.cos(step/steps*np.pi/2))) * L21_norm)
        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step()
        if verbose and step % 10 == 0:
            print()
            print(step, "delta.abs()", delta.abs().mean())
            print(step, "rotation", rotation.abs().mean(dim=0))
            print(step, "mean", y.mean(dim=1).abs().mean())
            print(step, "scale", y.flatten(1).std(1).mean())
    rot = compute_rotation_matrix_from_ortho6d(rotation)
    y = (x + delta)
    y = y - y.mean(dim=1, keepdim=True)
    y = y @ rot
    if args.save_itmd:
        return y, itmd, itmd_zt
    return y


def dda(args, model, x, mask, ind):
    # from chamfer_distance import ChamferDistance
    chamfer_dist = ChamferDistance()

    if isinstance(model, nn.DataParallel):
        model = model.module

    t = torch.full((x.size(0), 1), args.dda_steps / args.diffusion_steps).to(x.device)
    gamma_t = model.inflate_batch_array(model.gamma(t), x)
    alpha_t = model.alpha(gamma_t, x)
    sigma_t = model.sigma(gamma_t, x)

    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    eps = model.sample_noise(n_samples=x.size(0),
        n_nodes=x.size(1),
        node_mask=node_mask,
    )
    z_t = (x * alpha_t + eps * sigma_t).requires_grad_(True)

    for step in tqdm(range(args.dda_steps, 0, -1)):
        t = torch.full((x.size(0), 1), step / args.diffusion_steps).to(x.device)
        gamma_t = model.inflate_batch_array(model.gamma(t), x)
        alpha_t = model.alpha(gamma_t, x)
        sigma_t = model.sigma(gamma_t, x)

        t_m1 = torch.full((x.size(0), 1), (step - 1) / args.diffusion_steps).to(x.device)
        z_t_m1 = model.sample_p_zs_given_zt(t_m1, t, z_t, node_mask).detach()

        # content preservation with low-pass filtering
        pred_noise = model(z_t, t=t, node_mask=node_mask, phi=True).detach()
        x0_est = (z_t - pred_noise * sigma_t) / alpha_t
        dist1, dist2 = chamfer_dist(low_pass_filtering(x0_est, args.dda_lpf_method, args.dda_lpf_scale), low_pass_filtering(x, args.dda_lpf_method, args.dda_lpf_scale))
        # dist1, dist2, _, _ = chamfer_dist(low_pass_filtering(x0_est, args.dda_lpf_method, args.dda_lpf_scale), low_pass_filtering(x, args.dda_lpf_method, args.dda_lpf_scale))
        cd_loss = torch.mean(dist1) + torch.mean(dist2)
        grad = torch.autograd.grad(
            cd_loss,
            z_t,
            allow_unused=True,
        )[0]
        z_t = (z_t_m1 - args.dda_guidance_weight * grad).requires_grad_(True)
    return z_t


def ours(args, model, x, mask, ind, classifier):
    if isinstance(model, nn.DataParallel):
        model = model.module
    if isinstance(classifier, nn.DataParallel):
        classifier = classifier.module

    t = torch.full((x.size(0), 1), args.ours_steps / args.diffusion_steps).to(x.device)
    gamma_t = model.inflate_batch_array(model.gamma(t), x)
    alpha_t = model.alpha(gamma_t, x)
    sigma_t = model.sigma(gamma_t, x)

    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    eps = model.sample_noise(n_samples=x.size(0),
        n_nodes=x.size(1),
        node_mask=node_mask,
    )
    z_t = (x * alpha_t + eps * sigma_t).requires_grad_(True)

    for step in tqdm(range(args.ours_steps, 0, -1)):
        t = torch.full((x.size(0), 1), step / args.diffusion_steps).to(x.device)
        gamma_t = model.inflate_batch_array(model.gamma(t), x)
        alpha_t = model.alpha(gamma_t, x)
        sigma_t = model.sigma(gamma_t, x)

        t_m1 = torch.full((x.size(0), 1), (step - 1) / args.diffusion_steps).to(x.device)
        z_t_m1 = model.sample_p_zs_given_zt(t_m1, t, z_t, node_mask).detach()

        # content preservation
        pred_noise = model(z_t, t=t, node_mask=node_mask, phi=True).detach()
        x0_est = (z_t - pred_noise * sigma_t) / alpha_t
        penultimate_layer_loss = F.mse_loss(classifier.get_feature(x0_est), classifier.get_feature(x).detach())
        grad = torch.autograd.grad(
            penultimate_layer_loss,
            z_t,
            allow_unused=True,
        )[0]
        z_t = (z_t_m1 - args.ours_guidance_weight * grad).requires_grad_(True)
    return z_t


def forward_and_adapt(args, classifier, optimizer, diffusion_model, x, mask, ind):
    import time
    start = time.time()

    global EMA, mom_pre

    if isinstance(classifier, torch.nn.DataParallel):
        classifier = classifier.module

    # input adaptation
    if 'pre_trans' in args.method:
        x = pre_trans(args, diffusion_model, x, mask, ind)
    if 'dda' in args.method:
        x_ori = x.clone()
        x = dda(args, diffusion_model, x, mask, ind)
    if 'ours' in args.method:
        x = ours(args, diffusion_model, x, mask, ind, classifier)

    # model adaptation
    for _ in range(1, args.num_steps + 1):
        # batch normalization statistics update methods
        if 'dua' in args.method:
            mom_new = mom_pre * args.dua_decay_factor
            for m in classifier.modules():
                if args.batch_size == 1 and isinstance(m, nn.BatchNorm1d):
                    m.eval()
                elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.train()
                    m.momentum = mom_new + args.dua_min_mom
            mom_pre = mom_new
            _ = classifier(x)
        if 'bn_stats' in args.method:
            _ = classifier(x)

        # model parameter adaptation
        if set(['tent', 'sar', 'pl', 'memo', 'shot']).intersection(args.method):
            optimizer.zero_grad()
        if 'tent' in args.method:
            logits = classifier(x)
            loss = softmax_entropy(logits).mean()
            loss.backward()
        if 'sar' in args.method:
            logits = classifier(x)
            entropy_first = softmax_entropy(logits)
            filter_id = torch.where(entropy_first < args.sar_ent_threshold * np.log(logits.shape[-1]))
            entropy_first = entropy_first[filter_id]
            loss = entropy_first.mean()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            new_logits = classifier(x)
            entropy_second = softmax_entropy(new_logits)
            entropy_second = entropy_second[filter_id]
            filter_id = torch.where(entropy_second < args.sar_ent_threshold * np.log(logits.shape[-1]))
            loss_second = entropy_second[filter_id].mean()
            loss_second.backward()
            optimizer.second_step(zero_grad=True)

            EMA = 0.9 * EMA + (1 - 0.9) * loss_second.item() if EMA != None else loss_second.item()
            continue # we already call optimizer.first_step and optimizer.second_step
        if 'pl' in args.method: 
            logits = classifier(x)
            pseudo_label = torch.argmax(logits, dim=-1)
            loss = F.cross_entropy(logits, pseudo_label)
            loss.backward()
        if 'memo' in args.method:
            x_aug = get_augmented_input(x, args.memo_num_augs)
            logits = classifier(x_aug)
            loss = marginal_entropy(args, logits)
            loss.backward()
        if 'shot' in args.method:
            # pseudo-labeling
            feats = classifier.get_feature(x).detach()
            logits = classifier(x)
            probs = logits.softmax(dim=-1)
            centroids = (feats.T @ probs) / probs.sum(dim=0, keepdim=True)
            pseudo_labels = (F.normalize(feats, p=2, dim=-1) @ F.normalize(centroids, p=2, dim=0)).argmax(dim=-1)
            new_centroids = (feats.T @ F.one_hot(pseudo_labels, num_classes=logits.shape[-1]).float()) / pseudo_labels.sum(dim=0, keepdim=True)
            new_pseudo_labels = (F.normalize(feats, p=2, dim=-1) @ F.normalize(new_centroids, p=2, dim=0)).argmax(dim=-1)
            pl_loss = F.cross_entropy(logits, new_pseudo_labels)
            loss = args.shot_pl_loss_weight * pl_loss

            # entropy loss
            entropy_loss = softmax_entropy(logits).mean()
            loss += entropy_loss

            # divergence loss
            softmax_out = F.softmax(logits, dim=-1)
            msoftmax = softmax_out.mean(dim=0)
            div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            loss += div_loss

            loss.backward()
        if set(['tent', 'sar', 'pl', 'memo', 'shot']).intersection(args.method):
            optimizer.step()

    end = time.time()
    print(f"end: {end - start}")

    # output adaptation
    is_training = classifier.training
    if is_training:
        classifier.eval()

    if 'lame' in args.method:
        logits_after = batch_evaluation(args, classifier, x)
    elif 'dda' in args.method:
        logits_after = ((classifier(x_ori).softmax(dim=-1) + classifier(x).softmax(dim=-1)) / 2).log()
    else:
        logits_after = classifier(x)

    if is_training:
        classifier.train()
    return logits_after


def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.random_seed) # set seed

    ########## logging ##########
    io = logging.IOStream(args)
    io.cprint(args)
    create_folders(args)
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name + '_vis', 'project':
            'adapt', 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    ########## load dataset ##########
    if args.dataset.startswith('modelnet40c'):
        test_dataset = ModelNet40C(args, partition='test')
    elif args.dataset in ['modelnet', 'shapenet', 'scannet']:
        test_dataset = PointDA10(args=args, partition='test')
    elif args.dataset in ['synthetic', 'kinect', 'realsense']:
        test_dataset = GraspNet10(args=args, partition='test')
    else:
        raise ValueError('UNDEFINED DATASET')
    if args.scenario == "label_distribution_shift":
        print(f"args.scenario: {args.scenario}")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(test_dataset, imb_ratio=args.imb_ratio), shuffle=False, drop_last=False, num_workers=args.num_workers)
    else:
        shuffle = False if args.scenario == "temporally_correlated" else True
        print(f"shuffle: {shuffle}")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=False, num_workers=args.num_workers)

    ########## load diffusion model ##########
    if 'pre_trans' in args.method or 'dda' in args.method or 'ours' in args.method:
        model = get_model(args, device)
        if args.diffusion_dir is not None:
            model.load_state_dict(torch.load(args.diffusion_dir, map_location='cpu'))
        model = nn.DataParallel(model)
        model = model.to(device).eval()
    else:
        model = None

    ########## load classifier ##########
    # TODO: add model architectures
    if args.classifier == "DGCNN":
        classifier = models.DGCNNWrapper(args.dataset, output_channels=len(np.unique(test_dataset.label_list)))
    else:
        raise ValueError('UNDEFINED CLASSIFIER')
    classifier.load_state_dict(torch.load(args.classifier_dir, map_location='cpu'))
    classifier = nn.DataParallel(classifier)
    classifier = classifier.to(device).eval()

    global EMA, mom_pre
    EMA = None
    mom_pre = args.dua_mom_pre

    original_classifier = deepcopy(classifier)
    original_classifier.eval().requires_grad_(False)
    classifier = configure_model(args, classifier)
    params, _ = collect_params(args, classifier, train_params=args.params_to_adapt)
    optimizer = setup_optimizer(args, params)
    original_classifier_state, original_optimizer_state, _ = copy_model_and_optimizer(classifier, optimizer, None)

    # args.dataset = "modelnet40c_original"
    # args.dataset_dir = "../datasets/modelnet40_ply_hdf5_2048"
    # train_dataset = ModelNet40C(
    #     args, 'train'
    # )
    # batch_size = 16
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=1)
    # source_feature_list, source_label_list, source_unary_list = [], [], []

    all_gt_list, all_pred_before_list, all_pred_after_list = [], [], []
    for iter_idx, data in tqdm(enumerate(test_loader)):
        x = data[0].to(device)
        labels = data[1].to(device).flatten()

        if args.adv_attack:
            x = projected_gradient_descent(args, classifier, x, labels, F.cross_entropy, num_steps=10, step_size=4e-3, step_norm='inf', eps=0.16, eps_norm='inf')
        all_gt_list.extend(labels.cpu().tolist())

        # reset source model and optimizer
        if args.episodic or ("sar" in args.method and EMA != None and EMA < 0.2):
            classifier, optimizer, _ = load_model_and_optimizer(classifier, optimizer, None, original_classifier_state, original_optimizer_state, None)

        logits_before = original_classifier(x).detach()
        all_pred_before_list.extend(logits_before.argmax(dim=-1).cpu().tolist())

        logits_after = forward_and_adapt(args, classifier, optimizer, model, x, mask=None, ind=None).detach()
        all_pred_after_list.extend(logits_after.argmax(dim=-1).cpu().tolist())

        io.cprint(f"batch idx: {iter_idx + 1}/{len(test_loader)}\n")
        io.cprint(f"cumulative metrics before adaptation | acc: {accuracy_score(all_gt_list, all_pred_before_list):.4f}")
        io.cprint(f"cumulative metrics after adaptation | acc: {accuracy_score(all_gt_list, all_pred_after_list):.4f}")
    io.cprint(f"final metrics before adaptation | acc: {accuracy_score(all_gt_list, all_pred_before_list):.4f}")
    io.cprint(f"final metrics after adaptation | acc: {accuracy_score(all_gt_list, all_pred_after_list):.4f}")
    return accuracy_score(all_gt_list, all_pred_after_list)


def tune_tta_hparams(args):
    import itertools, random
    if 'tent' in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        hparams_to_search = [test_lr_list, num_steps_list]
        hparams_to_search_str = ['test_lr', 'num_steps']
    if 'lame' in args.method:
        lame_affinity_list = ['rbf', 'kNN', 'linear']
        lame_knn_list = [1, 3, 5, 10]
        lame_max_steps_list = [1, 10, 100]
        hparams_to_search = [lame_affinity_list, lame_knn_list, lame_max_steps_list]
        hparams_to_search_str = ['lame_affinity', 'lame_knn', 'lame_max_steps']
    if 'sar' in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        sar_ent_threshold_list = [0.2, 0.4, 0.6, 0.8]
        sar_eps_threshold_list = [0.01, 0.05, 0.1]
        hparams_to_search = [test_lr_list, num_steps_list, sar_ent_threshold_list, sar_eps_threshold_list]
        hparams_to_search_str = ['test_lr', 'num_steps', 'sar_ent_threshold', 'sar_eps_threshold']
    if 'pl' in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        hparams_to_search_str = ['test_lr', 'num_steps']
        hparams_to_search = [test_lr_list, num_steps_list]
    if 'memo' in args.method:
        test_lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
        num_steps_list = [1, 2]
        memo_num_augs_list = [16, 32, 64]
        hparams_to_search = [test_lr_list, num_steps_list, memo_num_augs_list]
        hparams_to_search_str = ['test_lr', 'num_steps', 'memo_num_augs']
    if 'dua' in args.method:
        num_steps_list = [1, 3, 5, 10]
        dua_decay_factor_list = [0.9, 0.94, 0.99]
        hparams_to_search = [num_steps_list, dua_decay_factor_list]
        hparams_to_search_str = ['num_steps', 'dua_decay_factor']
    if 'bn_stats' in args.method:
        bn_stats_prior_list = [0, 0.2, 0.4, 0.6, 0.8]
        hparams_to_search = [bn_stats_prior_list]
        hparams_to_search_str = ['bn_stats_prior']
    if 'shot' in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        shot_pl_loss_weight_list = [0, 0.1, 0.3, 0.5, 1]
        hparams_to_search = [test_lr_list, num_steps_list, shot_pl_loss_weight_list]
        hparams_to_search_str = ['test_lr', 'num_steps', 'shot_pl_loss_weight']
    if 'dda' in args.method:
        dda_guidance_weight_list = [3, 6, 9]
        dda_lpf_scale_list = [2, 4, 8]
        hparams_to_search = [dda_guidance_weight_list, dda_lpf_scale_list]
        hparams_to_search_str = ['dda_guidance_weight', 'dda_lpf_scale']

    hparam_space = list(itertools.product(*hparams_to_search))
    random.shuffle(hparam_space)

    io = logging.IOStream(args)
    io.cprint(args)
    io.cprint(f"hyperparameter search: {hparam_space}")
    create_folders(args)

    best_acc, best_hparam = 0, None
    for hparam_comb in hparam_space[:min(len(hparam_space), 10)]:
        for hparam_str, hparam in zip(hparams_to_search_str, hparam_comb):
            setattr(args, hparam_str, hparam)
        io.cprint(f"hparams_to_search_str: {hparams_to_search_str}")
        io.cprint(f"hparam: {hparam_comb}")
        test_acc = main(args)
        io.cprint(f"test_acc: {test_acc}")
        if test_acc > best_acc:
            io.cprint(f"new best acc!: {test_acc}")
            best_acc = test_acc
        best_hparam = hparam_comb
    io.cprint(f"best result hparam, test_acc: {best_hparam}, {best_acc}")


@torch.no_grad()
def vis(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.random_seed) # set seed

    ########## logging ##########
    io = logging.IOStream(args)
    io.cprint(args)
    create_folders(args)
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name + '_vis', 'project':
            'adapt', 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    ########## load dataset ##########
    if args.dataset.startswith('modelnet40c'):
        test_dataset = ModelNet40C(args, partition='test')
    elif args.dataset in ['modelnet', 'shapnet', 'scannet']:
        test_dataset = PointDA10(args=args, partition='test')
    elif args.dataset in ['synthetic', 'kinect', 'realsense']:
        test_dataset = GraspNet10(args=args, partition='test')
    else:
        raise ValueError('UNDEFINED DATASET')
    test_loader_vis = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, sampler=ImbalancedDatasetSampler(test_dataset))

    ########## load diffusion model ##########
    if 'pre_trans' in args.method or 'dda' in args.method or 'ours' in args.method:
        model = get_model(args, device)
        if args.diffusion_dir is not None:
            model.load_state_dict(torch.load(args.diffusion_dir, map_location='cpu'))
        model = nn.DataParallel(model)
        model = model.to(device).eval()
    else:
        model = None

    ########## load classifier ##########
    # TODO: add model architectures
    if args.classifier == "DGCNN":
        classifier = models.DGCNNWrapper(args.dataset, output_channels=len(np.unique(test_dataset.label_list)))
    else:
        raise ValueError('UNDEFINED CLASSIFIER')
    classifier.load_state_dict(torch.load(args.classifier_dir, map_location='cpu'))
    classifier = nn.DataParallel(classifier)
    classifier = classifier.to(device).eval()

    global EMA, mom_pre
    EMA = None
    mom_pre = args.dua_mom_pre

    original_classifier = deepcopy(classifier)
    original_classifier.eval().requires_grad_(False)
    classifier = configure_model(args, classifier)
    params, _ = collect_params(classifier, train_params=args.params_to_adapt)
    optimizer = setup_optimizer(args, params)
    original_classifier_state, original_optimizer_state, _ = copy_model_and_optimizer(classifier, optimizer, None)

    all_gt_list, all_pred_before_list, all_pred_after_list = [], [], []
    for batch_idx, data in tqdm(enumerate(test_loader_vis)):
        x = data[0].to(device)
        labels = data[1].to(device).flatten()
        mask = data[2].to(device)
        ind = data[3].to(device) # original indices for duplicated point
        io.cprint(f"ground truth labels: {labels}")

        if args.adv_attack:
            x = projected_gradient_descent(args, classifier, x, labels, F.cross_entropy, num_steps=10, step_size=4e-3, step_norm='inf', eps=0.16, eps_norm='inf')
        all_gt_list.extend(labels.cpu().tolist())

        rgbs_wMask = get_color(x.cpu().numpy(), mask=mask.bool().squeeze(-1).cpu().numpy())
        rgbs = get_color(x.cpu().numpy())
        # x_ori = x.detach()
        x_ori = ours(args, model, x, mask, ind, classifier)

        for b in range(len(x_ori)):
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
                            ])* 3
                            ).tolist(),
                            # "label": f'{labels[b]} {preds_label[b]} {preds_val[b]:.2f}',
                            # "label": f'{labels[b]} {preds_label[b]}',
                            "color": [123, 321, 111], # ???
                        }
                    ]
                ),
            })
            wandb.log({f'pc': obj3d}, step=b, commit=False)

            obj3d = wandb.Object3D({
                "type": "lidar/beta",
                "points": np.concatenate((x_ori[b].cpu().numpy().reshape(-1, 3),
                    rgbs_wMask[b]), axis=1),
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
                            ])*3).tolist(),
                            # "label": f'{labels[b]} {preds_label[b]} {preds_val[b]:.2f}',
                            # "label": f'{labels[b]} {preds_label[b]}',
                            "color": [123, 321, 111],
                        }
                    ]
                ),
            })
            wandb.log({f'masked pc': obj3d}, step=b, commit=False)
        break
    io.cprint("Visualization Done")



if __name__ == "__main__":
    args = parse_arguments()
    if 'eval' in args.mode:
        main(args)
    if 'vis' in args.mode:
        vis(args)
    if 'hparam_tune' in args.mode:
        tune_tta_hparams(args)