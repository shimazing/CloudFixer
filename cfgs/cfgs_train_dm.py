import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Diffusion for PC")
    parser.add_argument("--exp_name", type=str, default="outputs")
    parser.add_argument("--output_dir", type=str, default="debug_10")
    parser.add_argument("--dataset", type=str, default="modelnet40")
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument(
        "--model", type=str, default="transformer", choices=["transformer"]
    )
    parser.add_argument(
        "--probabilistic_model", type=str, default="diffusion", help="diffusion"
    )
    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    parser.add_argument("--diffusion_steps", type=int, default=500)
    parser.add_argument(
        "--diffusion_noise_schedule",
        type=str,
        default="polynomial_2",
        help="learned, cosine, linear",
    )
    parser.add_argument(
        "--diffusion_noise_precision",
        type=float,
        default=1e-5,
    )
    parser.add_argument("--diffusion_loss_type", type=str, default="l2", help="vlb, l2")
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_nodes", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dp", type=eval, default=True, help="True | False")
    parser.add_argument("--clip_grad", type=eval, default=True, help="True | False")
    parser.add_argument("--n_report_steps", type=int, default=1)
    parser.add_argument("--wandb_usr", type=str, default="unknown")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    parser.add_argument(
        "--online",
        type=bool,
        default=True,
        help="True = wandb online -- False = wandb offline",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument("--save_model", type=eval, default=True, help="save model")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of worker for the dataloader"
    )
    parser.add_argument("--test_epochs", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="")
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="Amount of EMA decay, 0 means off. A reasonable value" " is 0.999.",
    )
    parser.add_argument("--random_scale", action="store_true")
    parser.add_argument(
        "--include_charges", type=eval, default=True, help="include atom charge or not"
    )
    parser.add_argument("--jitter", type=eval, default=False)
    parser.add_argument(
        "--visualize_every_batch",
        type=int,
        default=1e8,
        help="Can be used to visualize multiple times per epoch",
    )
    parser.add_argument("--out_path", type=str, default="./exps")
    parser.add_argument("--accum_grad", type=int, default=1)
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="unit_std",
        choices=["unit_val", "unit_std", "unit_norm"],
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--test_ema", action="store_true")
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--no_zero_mean", action="store_true")
    parser.add_argument("--lr_gamma", default=1, type=float)
    parser.add_argument("--cls_uniform", default=True, type=eval)
    args = parser.parse_args()
    args.zero_mean = not args.no_zero_mean
    return args