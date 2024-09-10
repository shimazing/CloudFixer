import os
import argparse
import yaml

import torch



def parse_arguments():
    parser = argparse.ArgumentParser(description="CloudFixer")
    parser.add_argument("--mode", nargs="+", type=str, required=True)
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument("--hparam_save_dir", type=str, default="cfgs/hparams")
    parser.add_argument("--use_best_hparam", action="store_true", default=False)

    # experiments
    parser.add_argument("--out_path", type=str, default="outputs")
    parser.add_argument("--exp_name", type=str, default="adaptation")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of worker for the dataloader"
    )
    parser.add_argument(
        "--online",
        type=bool,
        default=True,
        help="True = wandb online -- False = wandb offline",
    )
    parser.add_argument("--wandb_usr", type=str, default="unknown")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")

    # dataset
    parser.add_argument("--dataset", type=str, default="modelnet40c_background_5")
    parser.add_argument("--dataset_dir", type=str, default="../datasets/modelnet40_c/")
    parser.add_argument("--adv_attack", type=eval, default=False)
    parser.add_argument("--scenario", type=str, default="normal")
    parser.add_argument("--imb_ratio", type=float, default=0)
    parser.add_argument("--rotate", type=eval, default=True)

    # classifier
    parser.add_argument("--classifier", type=str, default="DGCNN")
    parser.add_argument(
        "--classifier_dir",
        type=str,
        default="checkpoints/dgcnn_modelnet40_best_test.pth",
    )
    parser.add_argument("--cls_scale_mode", type=str, default="unit_norm")

    # method and hyperparameters
    parser.add_argument("--method", nargs="+", type=str, default=["pre_trans"])
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=4)

    # tta hyperparameters
    parser.add_argument("--episodic", type=eval, default=True)
    parser.add_argument("--test_optim", type=str, default="AdamW")
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--test_lr", type=float, default=1e-4)
    parser.add_argument("--params_to_adapt", nargs="+", type=str, default=["all"])
    parser.add_argument(
        "--lame_affinity", type=str, required=False, default="rbf"
    )  # for LAME
    parser.add_argument("--lame_knn", type=int, required=False, default=5)  # for LAME
    parser.add_argument(
        "--lame_max_steps", type=int, required=False, default=1
    )  # for LAME
    parser.add_argument("--sar_ent_threshold", type=float, default=0.4)  # for SAR
    parser.add_argument("--sar_eps_threshold", type=float, default=0.05)  # for SAR
    parser.add_argument(
        "--memo_num_augs", type=int, required=False, default=4
    )  # for MEMO
    parser.add_argument(
        "--memo_bn_momentum", type=eval, default=1 / 17
    )  # for memo, dua, ...
    parser.add_argument("--dua_mom_pre", type=float, default=0.1)
    parser.add_argument("--dua_decay_factor", type=float, default=0.94)
    parser.add_argument("--dua_min_mom", type=float, default=0.005)
    parser.add_argument("--bn_stats_prior", type=float, default=0)
    parser.add_argument("--shot_pl_loss_weight", type=float, default=0.3)
    parser.add_argument("--dda_steps", type=int, default=100)
    parser.add_argument("--dda_guidance_weight", type=float, default=6)
    parser.add_argument("--dda_lpf_method", type=str, default="fps")
    parser.add_argument("--dda_lpf_scale", type=float, default=4)

    # diffusion model
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument(
        "--probabilistic_model", type=str, default="diffusion", help="diffusion"
    )
    parser.add_argument(
        "--diffusion_dir",
        type=str,
        default="checkpoints/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy",
    )
    parser.add_argument("--diffusion_steps", type=int, default=500)
    parser.add_argument(
        "--diffusion_noise_schedule",
        type=str,
        default="polynomial_2",
        help="learned, cosine, linear",
    )
    parser.add_argument("--diffusion_noise_precision", type=float, default=1e-5)
    parser.add_argument("--diffusion_loss_type", type=str, default="l2", help="vlb, l2")
    parser.add_argument("--scale_mode", type=str, default="unit_std")
    parser.add_argument("--n_nodes", type=int, default=1024)
    parser.add_argument("--dp", type=eval, default=True, help="True | False")
    parser.add_argument("--accum_grad", type=int, default=1)
    parser.add_argument("--t", type=float, default=0.4)

    # cloudfixer hyperparameters
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument("--input_lr", type=float, default=1e-2)
    parser.add_argument("--n_update", default=30, type=int)
    parser.add_argument("--rotation", default=0.1, type=float)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--weighted_reg", type=eval, default=True)
    parser.add_argument("--reg_method", type=str, default="inv_dist")
    parser.add_argument("--pow", type=int, default=1)

    parser.add_argument("--warmup", default=0.2, type=float)
    parser.add_argument("--lam_l", type=float, default=0)
    parser.add_argument("--lam_h", type=float, default=0)
    parser.add_argument("--t_min", type=float, default=0.02)
    parser.add_argument("--t_len", type=float, default=0.1)

    parser.add_argument("--optim", type=str, default="adamax")
    parser.add_argument("--optim_end_factor", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n_iters_per_update", type=int, default=1)
    parser.add_argument("--subsample", type=int, default=2048)
    parser.add_argument("--denoising_thrs", type=int, default=0)
    parser.add_argument("--vote", type=int, default=1)
    args = parser.parse_args()
    if "eval" in args.mode:
        args.no_wandb = True
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if "modelnet40" in args.classifier_dir:
        source_dataset = "modelnet40c_original"
    elif "modelnet" in args.classifier_dir:
        source_dataset = "modelnet"
    elif "shapenet" in args.classifier_dir:
        source_dataset = "shapenet"
    elif "scannet" in args.classifier_dir:
        source_dataset = "scannet"
    yaml_parent_dir = os.path.join(
        args.hparam_save_dir, args.classifier, source_dataset
    )
    yaml_dir = os.path.join(yaml_parent_dir, f"{'_'.join(args.method)}.yaml")

    if args.use_best_hparam and os.path.exists(yaml_dir):
        hparam_dict = yaml.load(open(yaml_dir, "r"), Loader=yaml.FullLoader)
        for hparams_to_search_str, best_hparam in hparam_dict.items():
            setattr(args, hparams_to_search_str, best_hparam)
        print(f"load best hyperparameters: {hparam_dict=}")
    return args