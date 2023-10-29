import os
import random
import argparse
import wandb
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.cuda.amp import custom_bwd, custom_fwd

from data.dataloader import ModelNet40C, PointDA10, GraspNet10, ImbalancedDatasetSampler
from build_model import get_model
from dgcnn_modelnet40 import DGCNN as DGCNN_modelnet40
import utils
from utils_GAST.pc_utils_Norm import scale_to_unit_cube_torch, rotate_shape_tensor
import log
from visualizer import visualize_pclist


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
parser.add_argument('--resume', type=str,
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
parser.add_argument('--classifier', type=str,
        default='outputs/dgcnn_modelnet40_best_test.pth')
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
utils.create_folders(args)

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
if args.resume is not None:
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
model = model.to(device)
model.eval()
model_dp = torch.nn.DataParallel(model)

# TODO: add other classifiers with different datasets
# if args.dataset.startswith('modelnet40'):
#     classifier = DGCNN_modelnet40()
#     classifier.load_state_dict(torch.load(args.classifier, map_location='cpu')['model_state'])
#     classifier = torch.nn.DataParallel(classifier)
#     classifier.to(device).eval()
#     print("load classifier_modelnet40")
# TODO: add other classifiers with different datasets
classifier = DGCNN_modelnet40()
classifier.load_state_dict(torch.load(args.classifier, map_location='cpu')['model_state'])
classifier = torch.nn.DataParallel(classifier)
classifier.to(device).eval()


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


def split_set(dataset, domain='scannet', set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler



class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3

    return out


def compute_rotation_matrix_from_ortho6d(poses): # 6DRepNet (https://github.com/thohemp/6DRepNet/blob/385008b04f32713347e689b27b92067cba710c42/sixdrepnet/utils.py#L144)
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


def matching_loss(x, step, steps, t=None, w=None):
    if t is None:
        t = (args.t_min * min(1, step/args.denoising_thrs) + (1-min(1,
            step/args.denoising_thrs)) * max(args.t_min, args.t_max - 0.2)) + 0.2 * torch.rand(x.shape[0], 1).to(x.device)
    gamma_t = model.inflate_batch_array(model.gamma(t), x)
    alpha_t = model.alpha(gamma_t, x)
    sigma_t = model.sigma(gamma_t, x)

    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    eps = model.sample_noise(n_samples=x.size(0),
        n_nodes=x.size(1),
        node_mask=node_mask,
    )
    z_t = x*alpha_t + eps*sigma_t
    pred_noise = model_dp(z_t, t=t, node_mask=node_mask, phi=True)
    loss = (pred_noise - eps).pow(2).mean()
    return loss, z_t.detach().clone().cpu()


@torch.enable_grad()
def pre_trans(x, mask, ind, lr=args.lr, steps=args.n_update, verbose=True):
    delta = torch.nn.Parameter(torch.zeros_like(x))
    rotation = x.new_zeros((x.size(0), 6))
    rotation[:, 0] = 1
    rotation[:, 4] = 1
    rotation = torch.nn.Parameter(rotation)
    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)

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
        matching, zt = matching_loss(y, step+1, steps)
        if args.save_itmd > 0 and step % args.save_itmd == 0:
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


def softmax(x):
    max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def get_color(coords, corners=np.array([
                                    [-1, -1, -1],
                                    [-1, 1, -1],
                                    [-1, -1, 1],
                                    [1, -1, -1],
                                    [1, 1, -1],
                                    [-1, 1, 1],
                                    [1, -1, 1],
                                    [1, 1, 1]
                                ]) * 3,
            mask=None,
    ): # Visualization
    coords = np.array(coords) # batch x n_points x 3
    corners = np.array(corners) # n_corners x 3
    colors = np.array([
        [255, 0, 0],
        [255, 127, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [75, 0, 130],
        [143, 0, 255],
    ])

    dist = np.linalg.norm(coords[:, :, None, :] -
            corners[None, None, :, :], axis=-1)

    weight = softmax(-dist)[:, :, :, None] #batch x NUM_POINTS x n_corners x 1
    rgb = (weight * colors).sum(2).astype(int) # NUM_POINTS x 3
    if mask is not None:
        # mask : B x N
        # rgb : B x N x 3 (RGB)
        rgb[(~mask).nonzero()] = np.array([255, 255, 255])
    return rgb


def adapt(args):
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    with torch.no_grad():
        count = 0
        correct_count = [0]
        label_batch_list = []
        for iter_idx, data in tqdm(enumerate(test_loader)):
            x_ori = scale_to_unit_cube_torch(data[2].to(device))
            mask = data[3].to(device)
            ind = data[4].to(device) # ori ind for duplicated point
            print("get ori")
            labels = data[1].to(device).flatten()

            # is_ori : batch_size x 1024
            x = data[0].to(device)
            if args.pre_trans:
                x = pre_trans(x, mask, ind)
            count += len(x)
            t = args.t * x.new_ones((x.shape[0], 1), device=x.device)
            node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
            gamma_t = model.inflate_batch_array(model.gamma(t), x)
            alpha_t = model.alpha(gamma_t, x)
            sigma_t = model.sigma(gamma_t, x)

            t0 = torch.zeros_like(t)
            gamma_t0 = model.inflate_batch_array(model.gamma(t0), x)
            alpha_t0 = model.alpha(gamma_t0, x)
            sigma_t0 = model.sigma(gamma_t0, x)
            noise_t0 = model.sample_noise(n_samples=x.size(0),
                    n_nodes=x.size(1),
                    node_mask=node_mask,
                    )

            labels_list = [labels]
            x_edit_list = [x]
            if args.cls_scale_mode == 'unit_norm':
                print("scaling")
                x_edit_list = \
                    [rotate_shape_tensor(
                        scale_to_unit_cube_torch(x.clone().detach()),
                            'x', np.pi/2) for x in
                                x_edit_list] # undo scaling
            for k, x_edit in enumerate(x_edit_list):
                logits = classifier(x_edit)
                preds = logits["cls"].max(dim=1)[1] # argmax
                ori_preds = preds
                ori_probs = logits["cls"].softmax(dim=-1)

                correct_count[k] += (preds == labels).long().sum().item()

            io.cprint("ACC")
            for ck, cc in enumerate(correct_count):
                io.cprint(f'{ck} {cc/max(count, 1)*100}')
    io.cprint(args)


def vis(args):
    if 'vis' in args.mode:
        with torch.no_grad():
            for _, data in enumerate(test_loader_vis):
                labels = [test_dataset.idx_to_label[int(d)] for d in data[1]]
                print("GT", labels)
                x = data[0].to(device)
                mask = data[3].to(device)
                ind = data[4].to(device)
                if args.pre_trans:
                    if args.save_itmd:
                        x, itmd, itmd_zt = pre_trans(x, mask, ind)
                    else:
                        x = pre_trans(x, mask, ind)

                file_list = [f"imgs/batch_{i}.png" for i in range(len(x))]
                visualize_pclist(x, file_list, colorm=[24,107,239]) # specify color with colorm

                rgbs_wMask = get_color(x.cpu().numpy(),
                        mask=mask.bool().squeeze(-1).cpu().numpy())
                rgbs = get_color(x.cpu().numpy())
                t = args.t * x.new_ones((x.shape[0], 1), device=x.device) # 0~1
                node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
                gamma_t = model.inflate_batch_array(model.gamma(t), x)

                x_ori = x.detach()
                if args.cls_scale_mode == 'unit_norm':
                    logits = classifier(rotate_shape_tensor(scale_to_unit_cube_torch(x.clone().detach()), 'x', np.pi/2))
                else:
                    logits = classifier(x_ori.clone().detach().permute(0, 2, 1), ts=x_ori.new_zeros(len(x_ori)), activate_DefRec=False)
                preds = logits["cls"].max(dim=1)[1]
                preds_val = logits["cls"].softmax(-1).max(dim=1)[0]
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
                break
    print("Visualization Done")



if __name__ == "__main__":
    if 'eval' in args.mode:
        adapt(args)
    if 'vis' in args.mode:
        vis(args)