import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        optimizer = SAM(params=params, base_optimizer=getattr(torch.optim, args.test_optim), lr=args.test_lr, rho=args.sar_eps_threshold)
    else:
        optimizer = getattr(torch.optim, args.test_optim)(params, lr=args.test_lr)
    return optimizer



def configure_bn_layer(args, m):
    m.requires_grad_(True)
    if not isinstance(m, nn.BatchNorm1d) or args.batch_size > 1:
        m.track_running_stats = False
        m.running_mean = None
        m.running_var = None


def configure_model(args, model):
    if 'tent' in args.method:
        model.eval()
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                configure_bn_layer(args, m)
    if 'sar' in args.method:
        model.eval()
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                configure_bn_layer(args, m)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
    if 'pl' in args.method:
        model.eval()
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                configure_bn_layer(args, m)
    if 'memo' in args.method:
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                configure_bn_layer(args, m)
                m.momentum = args.memo_bn_momentum
    if 'shot' in args.method:
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                configure_bn_layer(args, m)
        m = list(model.modules())[-1]
        m.eval()
        m.requires_grad_(False)
    if 'dua' in args.method:
        model.eval()
        for m in model.modules():
            if args.batch_size == 1 and isinstance(m, nn.BatchNorm1d):
                m.eval()
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()
    if 'bn_stats' in args.method:
        model.eval()
        for m in model.modules():
            if args.batch_size == 1 and isinstance(m, nn.BatchNorm1d):
                m.eval()
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()
                if not args.bn_stats_prior: # do not use source statistics
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                else:
                    m.track_running_stats = True
                    m.momentum = 1 - args.bn_stats_prior
    return model


def collect_params(args, model, train_params):
    params, names = [], []
    for nm, m in model.named_modules():
        if 'all' in train_params:
            for np, p in m.named_parameters():
                p.requires_grad = True # for SHOT
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
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
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


def marginal_entropy(args, logits):
    per_sample_logits = []
    for i in range(0, len(logits), args.memo_num_augs):
        per_sample_logits.append(logits[i:i + args.memo_num_augs])
    per_sample_logits = torch.stack(per_sample_logits, dim=0)
    probs = per_sample_logits.softmax(dim=-1)
    probs = probs.mean(dim=1)
    return -(probs * torch.log(probs)).sum(dim=-1).mean()


def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)




class AffinityMatrix:
    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())



class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)
        return W



class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        return rbf



class linear_affinity(AffinityMatrix):
    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)
        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            break
        else:
            oldE = E
    return Y


def batch_evaluation(args, model, x):
    out = model(x).detach()
    unary = -torch.log(out.softmax(-1) + 1e-10)  # softmax the output
    # feats = F.normalize(model.get_feature(x), p=2, dim=-1).detach()
    feats = F.normalize(model(x, return_feature=True), p=2, dim=-1).detach()
    affinity = eval(f'{args.lame_affinity}_affinity')(sigma=1.0, knn=args.lame_knn)
    kernel = affinity(feats)
    Y = laplacian_optimization(unary, kernel, max_steps=args.lame_max_steps)
    return Y