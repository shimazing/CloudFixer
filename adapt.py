import argparse
from copy import deepcopy
from tqdm import tqdm
import wandb
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score
from chamfer_distance import ChamferDistance as chamfer_dist

chamfer_dist_fn = chamfer_dist()

from cfgs.cfgs_adapt import parse_arguments
from datasets.dataloader import *
from diffusion.build_model import get_model
from classifier import models
from utils import logging
from utils.utils import *
from utils.pc_utils import *
from utils.tta_utils import *


def dda(args, model, x, mask, ind):
    # from chamfer_distance import ChamferDistance
    # chamfer_dist = ChamferDistance()

    t = torch.full((x.size(0), 1), args.dda_steps / args.diffusion_steps).to(x.device)
    gamma_t = model.module.inflate_batch_array(model.module.gamma(t), x)
    alpha_t = model.module.alpha(gamma_t, x)
    sigma_t = model.module.sigma(gamma_t, x)

    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    eps = model.module.sample_noise(
        n_samples=x.size(0),
        n_nodes=x.size(1),
        node_mask=node_mask,
    )
    z_t = (x * alpha_t + eps * sigma_t).requires_grad_(True)

    for step in tqdm(range(args.dda_steps, 0, -1)):
        t = torch.full((x.size(0), 1), step / args.diffusion_steps).to(x.device)
        gamma_t = model.module.inflate_batch_array(model.module.gamma(t), x)
        alpha_t = model.module.alpha(gamma_t, x)
        sigma_t = model.module.sigma(gamma_t, x)

        t_m1 = torch.full((x.size(0), 1), (step - 1) / args.diffusion_steps).to(
            x.device
        )
        z_t_m1 = model(
            z_t, t, node_mask=node_mask, sample_p_zs_given_zt=True, s=t_m1
        ).detach()

        # content preservation with low-pass filtering
        pred_noise = model(z_t, t=t, node_mask=node_mask, phi=True).detach()
        x0_est = (z_t - pred_noise * sigma_t) / alpha_t
        dist1, dist2, _, _ = chamfer_dist_fn(
            low_pass_filtering(x0_est, args.dda_lpf_method, args.dda_lpf_scale),
            low_pass_filtering(x, args.dda_lpf_method, args.dda_lpf_scale),
        )
        cd_loss = torch.mean(dist1) + torch.mean(dist2)
        grad = torch.autograd.grad(
            cd_loss,
            z_t,
            allow_unused=True,
        )[0]
        z_t = (z_t_m1 - args.dda_guidance_weight * grad).requires_grad_(True)
    return z_t


@torch.enable_grad()
def cloudfixer(args, model, x, mask, ind, verbose=False):
    ######################## Scheduler ########################
    def get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        last_epoch=-1,
        end_factor=0,
    ):
        """
        Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return end_factor + max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

    ######################## End Scheduler ########################
    _, knn_dist_square_mean = knn(
        x.transpose(2, 1),
        k=args.knn,
        mask=(mask.squeeze(-1).bool()),
        ind=ind,
        return_dist=True,
    )
    knn_dist_square_mean = knn_dist_square_mean[torch.arange(x.size(0))[:, None], ind]
    weight = 1 / knn_dist_square_mean.pow(args.pow)
    if not args.weighted_reg:
        weight = torch.ones_like(weight)
    weight = weight / weight.sum(dim=-1, keepdim=True)  # normalize
    weight = weight * mask.squeeze(-1)
    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    delta = torch.nn.Parameter(torch.zeros_like(x))
    rotation = torch.nn.Parameter(x.new_zeros((x.size(0), 6)))
    rotation_base = x.new_zeros((x.size(0), 6))
    rotation_base[:, 0] = 1
    rotation_base[:, 4] = 1
    delta.requires_grad_(True)
    rotation.requires_grad_(True)

    optim = torch.optim.Adamax(
        [
            {"params": [delta], "lr": args.input_lr},
            {"params": [rotation], "lr": args.rotation},
        ],
        lr=args.input_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    scheduler = get_linear_schedule_with_warmup(
        optim,
        int(args.n_update * args.warmup),
        args.n_update,
        last_epoch=-1,
        end_factor=args.optim_end_factor,
    )

    iterator = tqdm(range(args.n_update)) if verbose else range(args.n_update)
    for iter in iterator:
        optim.zero_grad()
        t = args.t_min + args.t_len * torch.rand(x.shape[0], 1).to(x.device)
        t = (t * args.diffusion_steps).long().float() / args.diffusion_steps
        gamma_t = model.module.inflate_batch_array(model.module.gamma(t), x)
        alpha_t = model.module.alpha(gamma_t, x)  # batch_size x 1 x 1
        sigma_t = model.module.sigma(gamma_t, x)

        eps = torch.randn_like(x)
        x_trans = x + delta
        rot = compute_rotation_matrix_from_ortho6d(rotation + rotation_base)
        x_trans = x_trans @ rot
        z_loss = 0
        with torch.no_grad():
            x_trans_t = x_trans * alpha_t + sigma_t * eps
            _, x_trans_est = model(
                x_trans_t,
                phi=True,
                return_x0_est=True,
                t=t,
                node_mask=node_mask,
            )
        dist1, dist2, _, _ = chamfer_dist_fn(x_trans, x_trans_est)
        matching = dist1.mean() + dist2.mean()
        L2_norm = (delta.pow(2) * weight[:, :, None]).sum(dim=1).mean()
        norm = L2_norm * (
            args.lam_h * (1 - iter / args.n_update) + args.lam_l * iter / args.n_update
        )
        loss = matching + norm + z_loss
        loss.backward()
        optim.step()
        scheduler.step()
        if verbose and (iter) % 10 == 0:
            print("LR", scheduler.get_last_lr())
            print("rotation", (rotation_base + rotation).abs().mean(dim=0))
            print("delta", (delta).abs().mean().item())  # norm(2, dim=-1).mean()
            print(
                delta[mask.expand_as(delta) == 1].abs().mean().item(),
                delta[mask.expand_as(delta) == 0].abs().mean().item(),
            )
    rot = compute_rotation_matrix_from_ortho6d(rotation + rotation_base)
    x_trans = x + delta
    x_trans = x_trans @ rot
    if verbose:
        print("LR", scheduler.get_last_lr())
        print("rotation", (rotation_base + rotation).abs().mean(dim=0))
        print("delta", (delta).norm(2, dim=-1).mean())
    return x_trans


def forward_and_adapt(args, classifier, optimizer, diffusion_model, x, mask, ind):
    global EMA, mom_pre, adaptation_time

    import time

    start = time.time()

    # input adaptation
    if "dda" in args.method:
        x_ori = x.clone()
        x = dda(args, diffusion_model, x, mask, ind)
    if "cloudfixer" in args.method:
        x_list = [
            cloudfixer(args, diffusion_model, x, mask, ind).detach().clone()
            for v in range(args.vote)
        ]
        x = x_list[0]  # cloudfixer(args, diffusion_model, x, mask, ind)

    # model adaptation
    for _ in range(1, args.num_steps + 1):
        # batch normalization statistics update methods
        if "dua" in args.method:
            mom_new = mom_pre * args.dua_decay_factor
            for m in classifier.modules():
                if args.batch_size == 1 and isinstance(m, nn.BatchNorm1d):
                    m.eval()
                elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.train()
                    m.momentum = mom_new + args.dua_min_mom
            mom_pre = mom_new
            _ = classifier(x)
        if "bn_stats" in args.method:
            _ = classifier(x)
        # model parameter adaptation
        if set(["tent", "sar", "pl", "memo", "shot", "mate"]).intersection(args.method):
            optimizer.zero_grad()
        if "tent" in args.method:
            logits = classifier(x)
            loss = softmax_entropy(logits).mean()
            loss.backward()
        if "sar" in args.method:
            logits = classifier(x)
            entropy_first = softmax_entropy(logits)
            filter_id = torch.where(
                entropy_first < args.sar_ent_threshold * np.log(logits.shape[-1])
            )
            entropy_first = entropy_first[filter_id]
            loss = entropy_first.mean()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            new_logits = classifier(x)
            entropy_second = softmax_entropy(new_logits)
            entropy_second = entropy_second[filter_id]
            filter_id = torch.where(
                entropy_second < args.sar_ent_threshold * np.log(logits.shape[-1])
            )
            loss_second = entropy_second[filter_id].mean()
            loss_second.backward()
            optimizer.second_step(zero_grad=True)

            EMA = (
                0.9 * EMA + (1 - 0.9) * loss_second.item()
                if EMA != None
                else loss_second.item()
            )
            continue  # we already call optimizer.first_step and optimizer.second_step
        if "pl" in args.method:
            logits = classifier(x)
            pseudo_label = torch.argmax(logits, dim=-1)
            loss = F.cross_entropy(logits, pseudo_label)
            loss.backward()
        if "memo" in args.method:
            x_aug = get_augmented_input(x, args.memo_num_augs)
            logits = classifier(x_aug)
            loss = marginal_entropy(args, logits)
            loss.backward()
        if "shot" in args.method:
            # pseudo-labeling
            try:
                feats = classifier(x, return_feature=True).detach()
            except:
                feats = classifier(pc=x, return_feature=True).detach()

            logits = classifier(x)
            probs = logits.softmax(dim=-1)
            centroids = (feats.T @ probs) / probs.sum(dim=0, keepdim=True)
            pseudo_labels = (
                F.normalize(feats, p=2, dim=-1) @ F.normalize(centroids, p=2, dim=0)
            ).argmax(dim=-1)
            new_centroids = (
                feats.T @ F.one_hot(pseudo_labels, num_classes=logits.shape[-1]).float()
            ) / pseudo_labels.sum(dim=0, keepdim=True)
            new_pseudo_labels = (
                F.normalize(feats, p=2, dim=-1) @ F.normalize(new_centroids, p=2, dim=0)
            ).argmax(dim=-1)
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
        if "mate" in args.method:
            points = x.cuda()
            points = low_pass_filtering(
                points, method="fps", scale=max(int(points.shape[1] / 1024), 1)
            )
            points = [points for _ in range(48)]
            points = torch.squeeze(torch.vstack(points))
            loss_recon, _, _ = classifier(pts=points, classification_only=False)
            loss = loss_recon.mean()
            loss.backward()
        if set(["tent", "sar", "pl", "memo", "shot", "mate"]).intersection(args.method):
            optimizer.step()

    end = time.time()
    adaptation_time += end - start

    # output adaptation
    is_training = classifier.training
    if is_training:
        classifier.eval()

    if "lame" in args.method:
        logits_after = batch_evaluation(args, classifier, x)
    elif "dda" in args.method:
        logits_after = (
            (classifier(x_ori).softmax(dim=-1) + classifier(x).softmax(dim=-1)) / 2
        ).log()
    elif "mate" in args.method:
        logits_after = classifier(pts=x.cuda(), classifiation_only=True)
    elif "cloudfixer" in args.method:
        probs = 0
        for x_ in x_list:
            probs = probs + classifier(x_).softmax(dim=-1)
        probs /= args.vote
        print("voting", args.vote)
        logits_after = probs.log()
    elif "mate" in args.method:
        logits_after = classifier(pts=x.cuda(), classifiation_only=True)
    else:
        logits_after = classifier(x)

    if is_training:
        classifier.train()
    return logits_after


def main(args):
    global adaptation_time
    adaptation_time = 0

    device = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.random_seed)

    ########## logging ##########
    io = logging.IOStream(args)
    io.cprint(args)
    create_folders(args)

    ########## load dataset ##########
    if args.dataset.startswith("modelnet40c") and not args.classifier == "pointMAE":
        test_dataset = ModelNet40C(args, partition="test")
    elif args.dataset.startswith("modelnet40c") and args.classifier == "pointMAE":
        args.rotate = True
        test_dataset = ModelNet40C(args, partition="test")
    if args.dataset.startswith("modelnet40c"):
        test_dataset = ModelNet40C(args, partition="test")
    elif args.dataset in ["modelnet", "shapenet", "scannet"]:
        test_dataset = PointDA10(args=args, partition="test")
    else:
        raise ValueError("UNDEFINED DATASET")
    if args.scenario == "label_distribution_shift":
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=ImbalancedDatasetSampler(test_dataset, imb_ratio=args.imb_ratio),
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )
    else:
        shuffle = False if args.scenario == "temporally_correlated" else True
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=args.num_workers,
        )

    ########## load diffusion model ##########
    if (
        "pre_trans" in args.method
        or "dda" in args.method
        or "ours" in args.method
        or "cloudfixer" in args.method
    ):
        model = get_model(args, device)
        if args.diffusion_dir is not None:
            model.load_state_dict(torch.load(args.diffusion_dir, map_location="cpu"))
        model = nn.DataParallel(model)
        model = model.to(device).eval()
    else:
        model = None

    ########## load classifier ##########
    if args.classifier == "point2vec":
        from classifier.point2vec.models import Point2VecClassification
        from classifier.point2vec.utils.checkpoint import extract_model_checkpoint

        classifier = Point2VecClassification()
        classifier.cuda()
        classifier.setup()
        checkpoint = extract_model_checkpoint(args.classifier_dir)
        missing_keys, unexpected_keys = classifier.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    elif args.classifier == "DGCNN":
        classifier = models.DGCNNWrapper(
            args.dataset, output_channels=np.max(test_dataset.label_list) + 1
        )
        ckpt = torch.load(args.classifier_dir, map_location="cpu")
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        classifier.load_state_dict(ckpt)    
    elif args.classifier == "pointNeXt":
        from classifier.openpoints.utils import EasyConfig, load_checkpoint
        from classifier.openpoints.models import build_model_from_cfg

        cfg = EasyConfig()
        cfg.load("cfgs/modelnet40_pointnext.yaml")
        classifier = build_model_from_cfg(cfg.model)  # .to(cfg.rank)
        load_checkpoint(classifier, pretrained_path=args.classifier_dir)
    elif args.classifier == "pointMLP":
        from classifier.pointMLP.models import pointMLP

        classifier = pointMLP()
        classifier = torch.nn.DataParallel(classifier)
        classifier.load_state_dict(
            torch.load(
                args.classifier_dir,
                map_location="cpu",
            )["net"]
        )
    elif args.classifier == "pointMAE":
        from utils.config import cfg_from_yaml_file
        from utils import builder
        config = cfg_from_yaml_file("cfgs/cfgs_mate/tta/tta_modelnet.yaml")
        classifier = builder.model_builder(config.model)
        classifier.load_model_from_ckpt(args.classifier_dir, False)
        classifier.rotate = args.rotate
    else:
        raise ValueError("UNDEFINED CLASSIFIER")
    
    if "pointMLP" not in args.classifier:
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
    original_classifier_state, original_optimizer_state, _ = copy_model_and_optimizer(
        classifier, optimizer, None
    )

    all_gt_list, all_pred_before_list, all_pred_after_list = [], [], []
    for iter_idx, data in tqdm(enumerate(test_loader)):
        x = data[0].to(device)
        labels = data[1].to(device).flatten()
        mask = data[-2].to(device)
        ind = data[-1].to(device)

        if args.adv_attack:
            x = projected_gradient_descent(
                args,
                classifier,
                x,
                labels,
                F.cross_entropy,
                num_steps=10,
                step_size=4e-3,
                step_norm="inf",
                eps=0.16,
                eps_norm="inf",
            )
        all_gt_list.extend(labels.cpu().tolist())

        # reset source model and optimizer
        if args.episodic or ("sar" in args.method and EMA != None and EMA < 0.2):
            classifier, optimizer, _ = load_model_and_optimizer(
                classifier,
                optimizer,
                None,
                original_classifier_state,
                original_optimizer_state,
                None,
            )

        logits_before = original_classifier(x).detach()
        logits_after = forward_and_adapt(
            args, classifier, optimizer, model, x, mask=mask, ind=ind
        ).detach()

        all_pred_before_list.extend(logits_before.argmax(dim=-1).cpu().tolist())
        all_pred_after_list.extend(logits_after.argmax(dim=-1).cpu().tolist())

        io.cprint(f"batch idx: {iter_idx + 1}/{len(test_loader)}\n")
        io.cprint(f"cumulative metrics before adaptation | acc: {accuracy_score(all_gt_list, all_pred_before_list):.4f}")
        io.cprint(f"cumulative metrics after adaptation | acc: {accuracy_score(all_gt_list, all_pred_after_list):.4f}")

    io.cprint(f"final metrics before adaptation | acc: {accuracy_score(all_gt_list, all_pred_before_list):.4f}")
    io.cprint(f"final metrics after adaptation | acc: {accuracy_score(all_gt_list, all_pred_after_list):.4f}")
    io.cprint(f"final metrics before adaptation | macro recall: {recall_score(all_gt_list, all_pred_before_list, average='macro'):.4f}")
    io.cprint(f"final metrics after adaptation | macro recall: {recall_score(all_gt_list, all_pred_after_list, average='macro'):.4f}")
    io.cprint(f"total adaptation time: {adaptation_time}")
    return accuracy_score(all_gt_list, all_pred_after_list)


def tune_tta_hparams(args):
    import itertools, random
    yaml_parent_dir = os.path.join(args.hparam_save_dir, args.classifier, args.dataset)
    yaml_dir = os.path.join(yaml_parent_dir, f"{'_'.join(args.method)}.yaml")

    if "tent" in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        hparams_to_search = [test_lr_list, num_steps_list]
        hparams_to_search_str = ["test_lr", "num_steps"]
    if "lame" in args.method:
        lame_affinity_list = ["rbf", "kNN", "linear"]
        lame_knn_list = [1, 3, 5, 10]
        lame_max_steps_list = [1, 10, 100]
        hparams_to_search = [lame_affinity_list, lame_knn_list, lame_max_steps_list]
        hparams_to_search_str = ["lame_affinity", "lame_knn", "lame_max_steps"]
    if "sar" in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        sar_ent_threshold_list = [0.2, 0.4, 0.6, 0.8]
        sar_eps_threshold_list = [0.01, 0.05, 0.1]
        hparams_to_search = [
            test_lr_list,
            num_steps_list,
            sar_ent_threshold_list,
            sar_eps_threshold_list,
        ]
        hparams_to_search_str = [
            "test_lr",
            "num_steps",
            "sar_ent_threshold",
            "sar_eps_threshold",
        ]
    if "pl" in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        hparams_to_search_str = ["test_lr", "num_steps"]
        hparams_to_search = [test_lr_list, num_steps_list]
    if "memo" in args.method:
        test_lr_list = [1e-6, 1e-5, 1e-4, 1e-3]
        num_steps_list = [1, 2]
        memo_num_augs_list = [16, 32, 64]
        hparams_to_search = [test_lr_list, num_steps_list, memo_num_augs_list]
        hparams_to_search_str = ["test_lr", "num_steps", "memo_num_augs"]
    if "dua" in args.method:
        num_steps_list = [1, 3, 5, 10]
        dua_decay_factor_list = [0.9, 0.94, 0.99]
        hparams_to_search = [num_steps_list, dua_decay_factor_list]
        hparams_to_search_str = ["num_steps", "dua_decay_factor"]
    if "bn_stats" in args.method:
        bn_stats_prior_list = [0, 0.2, 0.4, 0.6, 0.8]
        hparams_to_search = [bn_stats_prior_list]
        hparams_to_search_str = ["bn_stats_prior"]
    if "shot" in args.method:
        test_lr_list = [1e-4, 1e-3, 1e-2]
        num_steps_list = [1, 3, 5, 10]
        shot_pl_loss_weight_list = [0, 0.1, 0.3, 0.5, 1]
        hparams_to_search = [test_lr_list, num_steps_list, shot_pl_loss_weight_list]
        hparams_to_search_str = ["test_lr", "num_steps", "shot_pl_loss_weight"]
    if "dda" in args.method:
        dda_guidance_weight_list = [3, 6, 9]
        dda_lpf_scale_list = [2, 4, 8]
        hparams_to_search = [dda_guidance_weight_list, dda_lpf_scale_list]
        hparams_to_search_str = ["dda_guidance_weight", "dda_lpf_scale"]

    hparam_space = list(itertools.product(*hparams_to_search))
    random.shuffle(hparam_space)

    io = logging.IOStream(args)
    io.cprint(args)
    io.cprint(f"hyperparameter search: {hparam_space}")
    create_folders(args)

    best_acc, best_hparam = 0, None
    for hparam_comb in hparam_space[: min(len(hparam_space), 30)]:
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

    yaml_parent_dir = os.path.join(args.hparam_save_dir, args.classifier, args.dataset)
    yaml_dir = os.path.join(yaml_parent_dir, f"{'_'.join(args.method)}.yaml")
    hparam_dict = dict(zip(hparams_to_search_str, best_hparam))
    if not os.path.exists(yaml_parent_dir):
        os.makedirs(yaml_parent_dir)

    with open(yaml_dir, "w") as f:
        yaml.dump(hparam_dict, f, sort_keys=False)
    io.cprint(f"best result hparam, test_acc: {best_hparam}, {best_acc}")


@torch.no_grad()
def vis(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    set_seed(args.random_seed)  # set seed

    ########## logging ##########
    io = logging.IOStream(args)
    io.cprint(args)
    create_folders(args)
    if args.no_wandb:
        mode = "disabled"
    else:
        mode = "online" if args.online else "offline"
    kwargs = {
        "entity": args.wandb_usr,
        "name": args.exp_name + "_vis",
        "project": "adapt",
        "config": args,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": mode,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")

    ########## load dataset ##########
    if args.dataset.startswith("modelnet40c"):
        test_dataset = ModelNet40C(args, partition="test")
    elif args.dataset in ["modelnet", "shapnet", "scannet"]:
        test_dataset = PointDA10(args=args, partition="test")
    else:
        raise ValueError("UNDEFINED DATASET")
    test_loader_vis = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        sampler=ImbalancedDatasetSampler(test_dataset),
    )

    ########## load diffusion model ##########
    if (
        "pre_trans" in args.method
        or "dda" in args.method
        or "ours" in args.method
        or "cloudfixer" in args.method
    ):
        model = get_model(args, device)
        if args.diffusion_dir is not None:
            model.load_state_dict(
                torch.load(args.diffusion_dir, map_location="cpu")
            )
        model = nn.DataParallel(model)
        model = model.to(device).eval()
    else:
        model = None

    ########## load classifier ##########
    # TODO: add model architectures
    if args.classifier == "point2vec":
        from classifier.point2vec.models import Point2VecClassification
        from classifier.point2vec.utils.checkpoint import extract_model_checkpoint

        classifier = Point2VecClassification()
        classifier.cuda()
        classifier.setup()
        checkpoint = extract_model_checkpoint(
            args.classifier_dir
        )
        missing_keys, unexpected_keys = classifier.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    elif args.classifier == "DGCNN":
        classifier = models.DGCNNWrapper(
            args.dataset, output_channels=len(np.unique(test_dataset.label_list))
        )
        classifier.load_state_dict(torch.load(args.classifier_dir, map_location="cpu"))
    elif args.classifier == "pointNeXt":
        from classifier.openpoints.utils import EasyConfig, load_checkpoint
        from classifier.openpoints.models import build_model_from_cfg

        cfg = EasyConfig()
        cfg.load("cfgs/modelnet40_pointnext.yaml")
        classifier = build_model_from_cfg(cfg.model)  # .to(cfg.rank)
        load_checkpoint(
            classifier,
            pretrained_path=args.classifier_dir,
        )
    elif args.classifier == "pointMLP":
        from classifier.pointMLP.models import pointMLP

        classifier = pointMLP()
        classifier = torch.nn.DataParallel(classifier)
        classifier.load_state_dict(
            torch.load(
                args.classifier_dir,
                map_location="cpu",
            )["net"]
        )
    else:
        raise ValueError("UNDEFINED CLASSIFIER")
    if "pointMLP" not in args.classifier:
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
    original_classifier_state, original_optimizer_state, _ = copy_model_and_optimizer(
        classifier, optimizer, None
    )

    all_gt_list, all_pred_before_list, all_pred_after_list = [], [], []
    for batch_idx, data in tqdm(enumerate(test_loader_vis)):
        x = data[0].to(device)
        labels = data[1].to(device).flatten()
        mask = data[2].to(device)
        ind = data[3].to(device)  # original indices for duplicated point
        io.cprint(f"ground truth labels: {labels}")

        if args.adv_attack:
            x = projected_gradient_descent(
                args,
                classifier,
                x,
                labels,
                F.cross_entropy,
                num_steps=10,
                step_size=4e-3,
                step_norm="inf",
                eps=0.16,
                eps_norm="inf",
            )
        all_gt_list.extend(labels.cpu().tolist())

        rgbs_wMask = get_color(
            x.cpu().numpy(), mask=mask.bool().squeeze(-1).cpu().numpy()
        )
        x_ori = x.detach()
        rgbs_ori = get_color(x_ori.cpu().numpy())
        # x_ori = ours(args, model, x, mask, ind, classifier)
        if "cloudfixer" in args.method:
            x = cloudfixer(args, model, x, mask, ind, verbose=True)
        # rgbs = get_color(x.cpu().numpy())
        rgbs = get_color(x.cpu().numpy(), mask=mask.bool().squeeze(-1).cpu().numpy())

        for b in range(len(x_ori)):
            obj3d = wandb.Object3D(
                {
                    "type": "lidar/beta",
                    "points": np.concatenate(
                        (x_ori[b].cpu().numpy().reshape(-1, 3), rgbs_ori[b]), axis=1
                    ),
                    "boxes": np.array(
                        [
                            {
                                "corners": (
                                    np.array(
                                        [
                                            [-1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [1, -1, -1],
                                            [1, 1, -1],
                                            [-1, 1, 1],
                                            [1, -1, 1],
                                            [1, 1, 1],
                                        ]
                                    )
                                    * 3
                                ).tolist(),
                                # "label": f'{labels[b]} {preds_label[b]} {preds_val[b]:.2f}',
                                # "label": f'{labels[b]} {preds_label[b]}',
                                "color": [123, 321, 111],  # ???
                            }
                        ]
                    ),
                }
            )
            wandb.log({f"pc": obj3d}, step=b, commit=False)

            obj3d = wandb.Object3D(
                {
                    "type": "lidar/beta",
                    "points": np.concatenate(
                        (x[b].cpu().numpy().reshape(-1, 3), rgbs[b]), axis=1
                    ),
                    "boxes": np.array(
                        [
                            {
                                "corners": (
                                    np.array(
                                        [
                                            [-1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [1, -1, -1],
                                            [1, 1, -1],
                                            [-1, 1, 1],
                                            [1, -1, 1],
                                            [1, 1, 1],
                                        ]
                                    )
                                    * 3
                                ).tolist(),
                                # "label": f'{labels[b]} {preds_label[b]} {preds_val[b]:.2f}',
                                # "label": f'{labels[b]} {preds_label[b]}',
                                "color": [123, 321, 111],  # ???
                            }
                        ]
                    ),
                }
            )
            wandb.log({f"adapted": obj3d}, step=b, commit=False)

            obj3d = wandb.Object3D(
                {
                    "type": "lidar/beta",
                    "points": np.concatenate(
                        (x_ori[b].cpu().numpy().reshape(-1, 3), rgbs_wMask[b]), axis=1
                    ),
                    "boxes": np.array(
                        [
                            {
                                "corners": (
                                    np.array(
                                        [
                                            [-1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [1, -1, -1],
                                            [1, 1, -1],
                                            [-1, 1, 1],
                                            [1, -1, 1],
                                            [1, 1, 1],
                                        ]
                                    )
                                    * 3
                                ).tolist(),
                                # "label": f'{labels[b]} {preds_label[b]} {preds_val[b]:.2f}',
                                # "label": f'{labels[b]} {preds_label[b]}',
                                "color": [123, 321, 111],
                            }
                        ]
                    ),
                }
            )
            wandb.log({f"masked pc": obj3d}, step=b, commit=False)
        break
    io.cprint("Visualization Done")


if __name__ == "__main__":
    args = parse_arguments()
    if "eval" in args.mode:
        main(args)
    if "vis" in args.mode:
        vis(args)
    if "hparam_tune" in args.mode:
        tune_tta_hparams(args)
