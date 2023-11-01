import copy

import torch
import torch.nn as nn

from utils.sam import SAM


def copy_model_and_optimizer(model, optimizer, scheduler):
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    if scheduler is not None:
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        return model_state, optimizer_state, scheduler_state
    else:
        return model_state, optimizer_state, None


def load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else:
        return model, optimizer, None


def setup_optimizer(args, params):
    if 'sar' in args.method:
        optimizer = SAM(params=params, base_optimizer=getattr(torch.optim, args.test_optim), lr=args.test_lr)
    else:
        optimizer = getattr(torch.optim, args.test_optim)(params, lr=args.test_lr)
    return optimizer


def collect_params(model, train_params):
    params, names = [], []
    for nm, m in model.named_modules():
        if 'all' in train_params:
            for np, p in m.named_parameters():
                p.requires_grad = True
                if not f"{nm}.{np}" in names:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if 'LN' in train_params:
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if 'BN' in train_params:
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if 'GN' in train_params:
            if isinstance(m, nn.GroupNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if "pretrain" in train_params:
            for np, p in m.named_parameters():
                if 'main_head' in f"{nm}.{np}":
                    continue
                params.append(p)
                names.append(f"{nm}.{np}")
        if "downstream" in train_params:
            for np, p in m.named_parameters():
                if not 'main_head' in f"{nm}.{np}":
                    continue
                params.append(p)
                names.append(f"{nm}.{np}")
    print(f"parameters to adapt: {names}")
    return params, names


def safe_log(x, ver):
    if ver == 1:
        return torch.log(x + 1e-5)
    elif ver == 2:
        return torch.log(x + 1e-7)
    elif ver == 3:
        return torch.clamp(torch.log(x), min=-100)
    else:
        raise ValueError("safe_log version is not properly defined !!!")


def softmax_entropy(x, dim=-1):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def softmax_diversity_regularizer(x): # shot
    x2 = x.softmax(-1).mean(0)  # [b, c] -> [c]
    return (x2 * safe_log(x2, ver=3)).sum()