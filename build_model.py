import torch

from diffusion.diffusion import DiffusionModel
from config_transformer import MODEL_CONFIGS, model_from_config


def get_model(args, device):
    if args.model == 'transformer':
        print("Transformer")
        net_dynamics = model_from_config(MODEL_CONFIGS['base40M-uncond'],
                device=device)
    else:
        raise ValueError('UNDEFINED DYNAMICS')
    if args.probabilistic_model == 'diffusion':
        vdm = DiffusionModel(
            dynamics=net_dynamics,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
        )
        return vdm #, nodes_dist
    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, model, ssl_loss=None):
    optim = torch.optim.AdamW(
        list(model.parameters()) + (list(ssl_loss.parameters()) if
            ssl_loss is not None else []),
        lr=args.lr, amsgrad=getattr(args, 'amsgrad', False),
        betas=(args.beta1, args.beta2) if hasattr(args, 'beta1') else (0.9, 0.999),
        weight_decay=getattr(args, 'wd', 1e-12)
    )
    return optim
