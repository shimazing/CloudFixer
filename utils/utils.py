import os
import random
import copy

import numpy as np
import torch

from utils.pc_utils import *



class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def set_seed(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass
    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path), map_location='cpu')
    model.eval()
    return model


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm


def softmax(x):
    max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def projected_gradient_descent(args, model, x, y, loss_fn, num_steps, step_size, step_norm, eps, eps_norm, y_target=None):
    rep_model = copy.deepcopy(model)
    rep_model.eval().to(x.device)

    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    # if args.cls_scale_mode == 'unit_norm':
    #     x_adv = rotate_shape_tensor(scale_to_unit_cube_torch(x.clone().detach().to(x.device)), 'x', np.pi/2)
    x_init = x_adv.clone().detach()
    targeted = y_target is not None
    num_channels = x.shape[1]

    for _ in range(num_steps):
        with torch.enable_grad():
            _x_adv = x_adv.clone().detach().requires_grad_(True).to(x.device)
            prediction = rep_model(_x_adv)
            loss = loss_fn(prediction, y_target if targeted else y)
            loss.backward()

        with torch.no_grad():
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1) \
                    .norm(step_norm, dim=-1)\
                    .view(-1, num_channels, 1, 1)
            if targeted:
                x_adv = x_adv - gradients
            else:
                x_adv = x_adv + gradients

        if eps_norm == 'inf':
            x_adv = torch.max(torch.min(x_adv, x_init + eps), x_init - eps)
        else:
            delta = x_adv - x
            mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps
            scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.view(-1, 1, 1)
            x_adv = x + delta

    # # unit_std normalization
    # x_adv = x_adv - x_adv.mean(dim=1, keepdim=True)
    # x_adv = x_adv / x_adv.view(x_adv.shape[0], -1).std(dim=1)[:, None, None]

    # if args.cls_scale_mode == 'unit_norm':
    #     x_adv = rotate_shape_tensor(x_adv, 'x', -np.pi/2)
    return x_adv.clone().detach()