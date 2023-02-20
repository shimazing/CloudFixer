import torch
from torch.distributions.categorical import Categorical

import numpy as np

from equivariant_diffusion.en_diffusion import DiffusionModel
try:
    from pointnet2.models.pointnet2_with_pcld_condition import PointNet2CloudCondition
    from pointnet2.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
except ImportError:
    print("pointnet unavalable")
import json
from config_transformer import MODEL_CONFIGS, model_from_config
try:
    from model.pvcnn_generation import PVCNN2Base
    class PVCNN2LargerBall(PVCNN2Base):
        sa_blocks = [
            ((32, 2, 32), (1024, 0.15, 32, (32, 64))),
            ((64, 3, 16), (256, 0.3, 32, (64, 128))),
            ((128, 3, 8), (64, 0.6, 32, (128, 256))),
            (None, (16, 1.2, 32, (256, 256, 512))),
        ]
        fp_blocks = [
            ((256, 256), (256, 3, 8)),
            ((256, 256), (256, 3, 8)),
            ((256, 128), (128, 2, 16)),
            ((128, 128, 64), (64, 2, 32)),
        ]

        def __init__(self, num_classes=3, embed_dim=64, use_att=True, dropout=0.1,
                extra_feature_channels=0, width_multiplier=1,
                     voxel_resolution_multiplier=1):
            super().__init__(
                num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
                dropout=dropout, extra_feature_channels=extra_feature_channels,
                width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
            )

        def forward(self, inputs, t, zero_mean=True):
            # inputs: (B,N,D)
            # coord: (B,N,D)
            inputs = inputs.permute(0,2,1)
            coord = super().forward(inputs, t)
            coord = coord.permute(0,2,1)
            if zero_mean:
                coord = coord - coord.mean(dim=1, keepdim=True)
            return coord


    class PVCNN2(PVCNN2Base):
        sa_blocks = [
            ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
            ((64, 3, 16), (256, 0.2, 32, (64, 128))),
            ((128, 3, 8), (64, 0.4, 32, (128, 256))),
            (None, (16, 0.8, 32, (256, 256, 512))),
        ]
        fp_blocks = [
            ((256, 256), (256, 3, 8)),
            ((256, 256), (256, 3, 8)),
            ((256, 128), (128, 2, 16)),
            ((128, 128, 64), (64, 2, 32)),
        ]

        def __init__(self, num_classes=3, embed_dim=64, use_att=True, dropout=0.1,
                extra_feature_channels=0, width_multiplier=1,
                     voxel_resolution_multiplier=1):
            super().__init__(
                num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
                dropout=dropout, extra_feature_channels=extra_feature_channels,
                width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
            )

        def forward(self, inputs, t, zero_mean=True):
            # inputs: (B,N,D)
            # coord: (B,N,D)
            inputs = inputs.permute(0,2,1)
            coord = super().forward(inputs, t)
            coord = coord.permute(0,2,1)
            if zero_mean:
                coord = coord - coord.mean(dim=1, keepdim=True)
            return coord
except:
    print("PVD unavalable")


def get_model(args, device):
    if args.model in ['pvd']:
        print("PVD")
        net_dynamics = PVCNN2()
    elif args.model == 'transformer':
        print("Transformer")
        net_dynamics = model_from_config(MODEL_CONFIGS['base40M-uncond'],
                device=device)
    else:
        print("PointNet")
        with open(args.dynamics_config) as f:
            data = f.read()
        config = json.loads(data)
        config = restore_string_to_list_in_a_dict(config)
        net_dynamics = PointNet2CloudCondition(config['pointnet_config'])

    if args.probabilistic_model == 'diffusion':
        vdm = DiffusionModel(
            dynamics=net_dynamics,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            zero_mean=not getattr(args, 'no_zero_mean', False),
            )

        return vdm #, nodes_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=getattr(args, 'amsgrad', False),
        betas=(args.beta1, args.beta2) if hasattr(args, 'beta1') else (0.9, 0.999),
        weight_decay=getattr(args, 'wd', 1e-12))

    return optim
