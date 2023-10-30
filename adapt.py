import argparse
import wandb
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataloader import ModelNet40C, PointDA10, GraspNet10, ImbalancedDatasetSampler
from build_model import get_model
from classification_model import models
from visualizer import visualize_pclist
from utils_GAST.pc_utils_Norm import scale_to_unit_cube_torch, rotate_shape_tensor
from utils_GAST import log
from utils import *


@torch.enable_grad()
def pre_trans(args, x, mask, ind, verbose=True):
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

    def matching_loss(x, step, t=None, w=None):
        if t is None:
            t = (args.t_min * min(1, step/args.denoising_thrs) + (1 - min(1, step/args.denoising_thrs)) * max(args.t_min, args.t_max - 0.2)) + 0.2 * torch.rand(x.shape[0], 1).to(x.device)

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

    delta = torch.nn.Parameter(torch.zeros_like(x))
    rotation = x.new_zeros((x.size(0), 6))
    rotation[:, 0] = 1
    rotation[:, 4] = 1
    rotation = torch.nn.Parameter(rotation)

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
        matching, zt = matching_loss(y, step+1)
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


def adapt(args):
    set_seed(args.random_seed)

    count = 0
    correct_count = [0]
    for iter_idx, data in tqdm(enumerate(test_loader)):
        x = data[0].to(device)
        labels = data[1].to(device).flatten()
        mask = data[2].to(device)
        ind = data[3].to(device) # original indices for duplicated point

        if args.pre_trans:
            x = pre_trans(args, x, mask, ind)
        count += len(x)
        
        x_edit_list = [x]
        if args.cls_scale_mode == 'unit_norm':
            x_edit_list = [rotate_shape_tensor(scale_to_unit_cube_torch(x.clone().detach()), 'x', np.pi/2) for x in x_edit_list]
        for k, x_edit in enumerate(x_edit_list):
            logits = classifier(x_edit)
            preds = logits.max(dim=1)[1] # argmax
            correct_count[k] += (preds == labels).long().sum().item()

        io.cprint("ACC")
        for ck, cc in enumerate(correct_count):
            io.cprint(f'{ck} {cc/max(count, 1)*100}')
    io.cprint(args)


@torch.no_grad()
def vis(args):
    for _, data in enumerate(test_loader_vis):
        x = data[0].to(device)
        labels = [test_dataset.idx_to_label[int(d)] for d in data[1]]
        print("GT", labels)
        mask = data[2].to(device)
        ind = data[3].to(device)
        if args.pre_trans:
            x = pre_trans(x, mask, ind)

        # new visualization code: you can specify color with colorm
        visualize_pclist(x, [f"imgs/batch_{i}.png" for i in range(len(x))], colorm=[24,107,239])

        rgbs_wMask = get_color(x.cpu().numpy(),
                mask=mask.bool().squeeze(-1).cpu().numpy())
        rgbs = get_color(x.cpu().numpy())

        x_ori = x.detach()
        if args.cls_scale_mode == 'unit_norm':
            logits = classifier(rotate_shape_tensor(scale_to_unit_cube_torch(x.clone().detach()), 'x', np.pi/2))
        else:
            logits = classifier(x_ori.clone().detach().permute(0, 2, 1), ts=x_ori.new_zeros(len(x_ori)), activate_DefRec=False)
        preds = logits.max(dim=1)[1]
        preds_val = logits.softmax(-1).max(dim=1)[0]
        preds_label = [test_dataset.idx_to_label[int(d)] for d in preds]
        print("ori", preds_label)
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
                            "label": f'{labels[b]} {preds_label[b]} {preds_val[b]:.2f}',
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
                            "label": f'{labels[b]} {preds_label[b]} {preds_val[b]:.2f}',
                            "color": [123, 321, 111], # ???
                        }
                    ]
                ),
            })
            wandb.log({f'masked pc': obj3d}, step=b, commit=False)
        break # TODO: remove break?
    print("Visualization Done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CloudFixer')
    parser.add_argument('--exp_name', type=str, default='adaptation')
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                        help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='learned, cosine, linear')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                        )
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='vlb, l2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_nodes', type=int, default=1024)
    parser.add_argument('--dp', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--wandb_usr', type=str, default='unknown')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb') # TODO
    parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    parser.add_argument('--diffusion_dir', type=str,
            default='outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy')
    parser.add_argument('--out_path', type=str, default='./exps')
    parser.add_argument('--knn', type=int, default=20)
    parser.add_argument('--accum_grad', type=int, default=1)
    parser.add_argument('--t', type=float, default=0.4)
    parser.add_argument('--scale_mode', type=str, default='unit_std')
    parser.add_argument('--cls_scale_mode', type=str, default='unit_norm')
    parser.add_argument('--mode', nargs='+', type=str, default=['eval'])
    parser.add_argument('--dataset', type=str, default='modelnet40c_background_5')
    parser.add_argument('--dataset_dir', type=str, default='../datasets/modelnet40_c/')
    parser.add_argument('--classifier', type=str, default='DGCNN')
    parser.add_argument('--classifier_dir', type=str, default='outputs/dgcnn_modelnet40_best_test.pth')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--pre_trans', action='store_true')
    parser.add_argument('--weighted_reg', type=eval, default=False)
    parser.add_argument('--n_update', default=400, type=int)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lam_h', type=float, default=0)
    parser.add_argument('--lam_l', type=float, default=0)
    parser.add_argument('--t_max', type=float, default=0.2)
    parser.add_argument('--t_min', type=float, default=0.02)
    parser.add_argument('--optim_end_factor', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--n_iters_per_update', type=int, default=1)
    #parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adamax')
    parser.add_argument('--subsample', type=int, default=1024)
    parser.add_argument('--pow', type=int, default=1)
    parser.add_argument('--denoising_thrs', type=int, default=100)
    parser.add_argument('--corrupt_ori', type=eval, default=False)
    parser.add_argument('--save_itmd', type=int, default=0)
    args = parser.parse_args()
    if 'eval' in args.mode:
        args.no_wandb = True
        args.save_itmd = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    io = log.IOStream(args)

    if args.dataset.startswith('modelnet40c'):
        test_dataset = ModelNet40C(args, partition='test')
    elif args.dataset in ['modelnet', 'shapnet', 'scannet']:
        test_dataset = PointDA10(args=args, partition='test')
    elif args.dataset in ['synthetic', 'kinect', 'realsense']:
        test_dataset = GraspNet10(args=args, partition='test')
    else:
        raise ValueError('UNDEFINED DATASET')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader_vis = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, sampler=ImbalancedDatasetSampler(test_dataset))

    io.cprint(args)
    create_folders(args)

    # Wandb config
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    kwargs = {'entity': args.wandb_usr, 'name': args.exp_name + '_vis', 'project':
            'adapt', 'config': args,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    model = get_model(args, device)
    if args.diffusion_dir is not None:
        model.load_state_dict(torch.load(args.diffusion_dir, map_location='cpu'))
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    # TODO: add other classifiers with different datasets
    if args.classifier == "DGCNN":
        class Args:
            def __init__(self):
                self.k = 20
                self.emb_dims = 1024
                self.dropout = 0.5
                self.leaky_relu = 1
        classifier_args = Args()
        classifier = models.DGCNN(args=classifier_args, output_channels=len(np.unique(test_dataset.label_list)))
    else:
        raise ValueError('UNDEFINED CLASSIFIER')
    classifier.load_state_dict(torch.load(args.classifier_dir, map_location='cpu'))
    classifier = torch.nn.DataParallel(classifier)
    classifier.to(device).eval()

    if 'eval' in args.mode:
        adapt(args)
    if 'vis' in args.mode:
        vis(args)