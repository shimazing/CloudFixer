# # Rdkit import should be first, do not move it
# try:
#     from rdkit import Chem
# except ModuleNotFoundError:
#     pass

# import argparse
# import wandb
# from tqdm import tqdm

# import numpy as np
# import torch
# import torch.nn as nn

# from data.dataloader import ModelNet40C, PointDA10, GraspNet10, ImbalancedDatasetSampler
# from build_model import get_optim, get_model
# from train_test import train_epoch, test
# from utils import uc_utils, log
# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
# import utils



# parser = argparse.ArgumentParser(description='CloudFixer')
# parser.add_argument('--exp_name', type=str, default='debug_10')
# parser.add_argument('--model', type=str, default='pointnet',
#                     help='our_dynamics | schnet | simple_dynamics | '
#                          'kernel_dynamics | egnn_dynamics |gnn_dynamics | '
#                          'pointnet')
# parser.add_argument('--probabilistic_model', type=str, default='diffusion',
#                     help='diffusion')

# # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
# parser.add_argument('--diffusion_steps', type=int, default=500)
# parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
#                     help='learned, cosine, linear')
# parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
#                     )
# parser.add_argument('--diffusion_loss_type', type=str, default='l2',
#                     help='vlb, l2')

# parser.add_argument('--n_epochs', type=int, default=200)
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--n_nodes', type=int, default=1024)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--brute_force', type=eval, default=False,
#                     help='True | False')
# parser.add_argument('--actnorm', type=eval, default=True,
#                     help='True | False')
# parser.add_argument('--break_train_epoch', type=eval, default=False,
#                     help='True | False')
# parser.add_argument('--dp', type=eval, default=True,
#                     help='True | False')
# parser.add_argument('--condition_time', type=eval, default=True,
#                     help='True | False')
# parser.add_argument('--clip_grad', type=eval, default=True,
#                     help='True | False')
# parser.add_argument('--trace', type=str, default='hutch',
#                     help='hutch | exact')
# parser.add_argument('--ode_regularization', type=float, default=1e-3)
# parser.add_argument('--n_report_steps', type=int, default=1)
# parser.add_argument('--wandb_usr', type=str, default='mazing')
# parser.add_argument('--no_wandb', action='store_true', help='Disable wandb',
#         default=True)
# parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--save_model', type=eval, default=True,
#                     help='save model')
# parser.add_argument('--generate_epochs', type=int, default=1,
#                     help='save model')
# parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
# parser.add_argument('--test_epochs', type=int, default=10)
# parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
# parser.add_argument("--conditioning", nargs='+', default=[],
#                     help='arguments : homo | lumo | alpha | gap | mu | Cv' )
# parser.add_argument('--resume', type=str,
#         default="outputs/pointnet/generative_model_last.npy", #required=True,
#                     #help='outputs/unit_val_shapenet_pointnet_resume/generative_model_last.npy'
#                     )
# parser.add_argument('--dynamics_config', type=str,
#         default='pointnet2/exp_configs/mvp_configs/config_standard_ori.json')
# parser.add_argument('--start_epoch', type=int, default=0,
#                     help='')
# parser.add_argument('--augment_noise', type=float, default=0)
# parser.add_argument('--n_stability_samples', type=int, default=500,
#                     help='Number of samples to compute the stability')
# parser.add_argument('--normalize_factors', type=eval, default=[1, 1, 1],
#                     help='normalize factors for [x, categorical, integer]')
# parser.add_argument('--remove_h', action='store_true')
# parser.add_argument('--include_charges', type=eval, default=True,
#                     help='include atom charge or not')
# parser.add_argument('--jitter', type=eval, default=False)
# parser.add_argument('--visualize_every_batch', type=int, default=1e8,
#                     help="Can be used to visualize multiple times per epoch")
# parser.add_argument('--normalization_factor', type=float, default=1,
#                     help="Normalize the sum aggregation of EGNN")
# parser.add_argument('--aggregation_method', type=str, default='mean',
#                     help='"sum" or "mean"')
# parser.add_argument('--out_path', type=str, default='./exps')
# parser.add_argument('--knn', type=int, default=32)
# parser.add_argument('--accum_grad', type=int, default=1)
# parser.add_argument('--t', type=float, default=0.4)
# parser.add_argument('--K', type=int, default=1)
# parser.add_argument('--voting', type=str, default='hard', choices=['hard', 'soft'])
# parser.add_argument('--accum_edit', action='store_true', help='Disable wandb') # TODO
# parser.add_argument('--scale_mode', type=str, default='unit_std')
# parser.add_argument('--scale', type=float, default=1)
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
# parser.add_argument('--output_pts', type=int, default=512)
# parser.add_argument('--guidance_scale', type=float, default=0)
# parser.add_argument('--mode', nargs='+', type=str, default=['train'])
# parser.add_argument('--dataset', type=str, default='shapenet')
# parser.add_argument('--keep_sub', action='store_true', help='Disable wandb') # TODO
# parser.add_argument('--n_subsample', type=int, default=64)
# parser.add_argument('--classifier', type=str,
#     default='../GAST/experiments/GAST_balanced_unitnorm_randomscale0.2/model.ptdgcnn')
# parser.add_argument('--self_ensemble', action='store_true')
# parser.add_argument('--time_cond', type=eval, default=True)
# parser.add_argument('--fc_norm', action='store_true')
# parser.add_argument('--no_zero_mean', action='store_true', default=True)
# parser.add_argument('--input_transform', action='store_true')
# parser.add_argument('--bn', type=str, default='bn')
# parser.add_argument('--gn', type=eval, default=False)
# parser.add_argument('--use_ori', type=eval, default=False)
# parser.add_argument('--cls_classifier', type=str,
#     default='outputs/SDist_0.4_fc_normFalsebngnFalse_shapenet2scannet_withDMFalse_dmpvd_timecondTrue_clf_guidanceFalse100.0_augTrueFalseTrueTrue_useOriFalse_epochs100_clFalse1024False_byol_step2/best.pt')


# def softmax(x):
#     max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
#     e_x = np.exp(x - max) #subtracts each row with its max value
#     sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
#     f_x = e_x / sum
#     return f_x


# def get_color(coords, corners=np.array([
#                                     [-1, -1, -1],
#                                     [-1, 1, -1],
#                                     [-1, -1, 1],
#                                     [1, -1, -1],
#                                     [1, 1, -1],
#                                     [-1, 1, 1],
#                                     [1, -1, 1],
#                                     [1, 1, 1]
#                                 ]) * 2,
#     ):
#     coords = np.array(coords) # batch x n_points x 3
#     corners = np.array(corners) # n_corners x 3
#     colors = np.array([
#         [255, 0, 0],
#         [255, 127, 0],
#         [255, 255, 0],
#         [0, 255, 0],
#         [0, 255, 255],
#         [0, 0, 255],
#         [75, 0, 130],
#         [143, 0, 255],
#     ])

#     dist = np.linalg.norm(coords[:, :, None, :] -
#             corners[None, None, :, :], axis=-1)

#     weight = softmax(-dist)[:, :, :, None] #batch x NUM_POINTS x n_corners x 1
#     rgb = (weight * colors).sum(2).astype(int) # NUM_POINTS x 3
#     return rgb


# args = parser.parse_args()

# io = log.IOStream(args)


# def split_set(dataset, domain='scannet', set_type="source"):
#     """
#     Input:
#         dataset
#         domain - modelnet/shapenet/scannet
#         type_set - source/target
#     output:
#         train_sampler, valid_sampler
#     """
#     train_indices = dataset.train_ind
#     val_indices = dataset.val_ind
#     unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
#     io.cprint("Occurrences count of classes in " + set_type + " " + domain +
#               " train part: " + str(dict(zip(unique, counts))))
#     unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
#     io.cprint("Occurrences count of classes in " + set_type + " " + domain +
#               " validation part: " + str(dict(zip(unique, counts))))
#     # Creating PT data samplers and loaders:
#     train_sampler = SubsetRandomSampler(train_indices)
#     valid_sampler = SubsetRandomSampler(val_indices)
#     return train_sampler, valid_sampler


# if args.n_nodes == 1024:
#     # train
#     random_remove = True
#     p_keep = (0.3, 1)
#     dataset_src = ShapeNet(io, './data', 'train', jitter=args.jitter,
#             scale=args.scale,
#             scale_mode=args.scale_mode,
#             random_scale=False,
#             random_rotation=True, zero_mean=not args.no_zero_mean,
#             random_remove=random_remove, p_keep=p_keep,
#             )
#     dataset_tgt = ScanNet(io, './data', 'train', jitter=args.jitter,
#             scale=args.scale,
#             scale_mode=args.scale_mode,
#             random_scale=False,
#             random_rotation=True, zero_mean=not args.no_zero_mean)
#     # val
#     dataset_src_val = ShapeNet(io, './data', 'val', jitter=args.jitter,
#             scale=args.scale,
#             scale_mode=args.scale_mode,
#             random_scale=False,
#             random_rotation=False, zero_mean=not args.no_zero_mean,
#             random_remove=random_remove, p_keep=p_keep,
#             )
#     dataset_tgt_val = ScanNet(io, './data', 'test', jitter=args.jitter,
#             scale=args.scale,
#             scale_mode=args.scale_mode,
#             random_scale=False,
#             random_rotation=False, zero_mean=not args.no_zero_mean)

#     #if args.dataset == 'scannet':
#     #    test_dataset = ScanNet(io, './data', 'test', jitter=args.jitter,
#     #        scale=args.scale,
#     #        scale_mode=args.scale_mode,
#     #        random_rotation=False) # for classification
#     #elif args.dataset == 'modelnet':
#     #    test_dataset = ModelNet(io, './data', 'test', jitter=args.jitter,
#     #        scale=args.scale,
#     #        scale_mode=args.scale_mode,
#     #        random_rotation=False) # for classification
#     #elif args.dataset == 'shapenet':
#     #    test_dataset = ShapeNet(io, './data', 'test', jitter=args.jitter,
#     #        scale=args.scale,
#     #        scale_mode=args.scale_mode,
#     #        random_rotation=False) # for classification

#     # TODO jitter??!!
#     #train_dataset_sampler, val_dataset_sampler = split_set(dataset_,
#     #    domain='shapenet')

#     train_loader_src = DataLoader(dataset_src, batch_size=args.batch_size,
#             sampler=None, #train_dataset_sampler,
#             drop_last=True, num_workers=16)
#     train_loader_tgt = DataLoader(dataset_tgt, batch_size=args.batch_size,
#             sampler=None, #train_dataset_sampler,
#             drop_last=True, num_workers=16)
#     train_loader_tgt_iter = iter(train_loader_tgt)

#     val_loader_src = DataLoader(dataset_src_val, batch_size=args.batch_size,
#             sampler=None, #train_dataset_sampler,
#             drop_last=False, num_workers=args.num_workers)
#     val_loader_tgt = DataLoader(dataset_tgt_val, batch_size=args.batch_size,
#             sampler=None, #train_dataset_sampler,
#             drop_last=False, num_workers=args.num_workers)

#     #val_loader = DataLoader(dataset_, batch_size=args.batch_size,
#     #        sampler=val_dataset_sampler) #, drop_last=True)
#     #test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
#     #        shuffle=False, drop_last=False)
#     #test_loader_vis = DataLoader(test_dataset, batch_size=args.batch_size,
#     #        shuffle=False, drop_last=False,
#     #        sampler=ImbalancedDatasetSampler(test_dataset))

# else: # 2048
#     train_dset = ShapeNetCore(
#         path='data/shapenet.hdf5', #args.dataset_path,
#         cates=['airplane'], #args.categories,
#         split='train',
#         scale_mode='shape_unit', #args.scale_mode,
#     )
#     val_dset = ShapeNetCore(
#         path='data/shapenet.hdf5', #args.dataset_path,
#         cates=['airplane'], #args.categories,
#         split='val',
#         scale_mode='shape_unit', #args.scale_mode,
#     )

#     train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
#     val_loader = DataLoader(val_dset, batch_size=2*args.batch_size, shuffle=False)


# args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

# args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if args.cuda else "cpu")
# dtype = torch.float32

# # Wandb config
# args.no_wandb = 'preprocess' not in args.mode
# if args.no_wandb:
#     mode = 'disabled'
# else:
#     mode = 'online' if args.online else 'offline'
# kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project':
#         'domain clf', 'config': args,
#           'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
# wandb.init(**kwargs)
# wandb.save('*.txt')

# # alpha, sigma 계산을 위해
# time_cond = args.time_cond
# if time_cond:
#     model = get_model(args, device) #dataloaders['train'])
#     model = model.to(device)

# args.model = 'dgcnn'
# classifier = DGCNN(args).to(device).eval()
# classifier = torch.nn.DataParallel(classifier)
# if args.bn == 'bn':
#     from sync_batchnorm import convert_model
#     classifier = convert_model(classifier).to(device)
# optim = get_optim(args, classifier)
# if 'eval' in args.mode or 'preprocess' in args.mode:
#     classifier.load_state_dict(
#         torch.load(args.resume,  #'outputs/domain_classifier_DGCNN_shape_scan_timecondGN_fullt.pt',
#             map_location='cpu'))
#     print("Resume")
#     classifier.eval()

# args.nregions = 3  # (default value)
# cls_classifier = DGCNN(args).to(device).eval()
# cls_classifier = torch.nn.DataParallel(cls_classifier)
# if args.bn == 'bn':
#     from sync_batchnorm import convert_model
#     cls_classifier = convert_model(cls_classifier).to(device)
# if 'preprocess' in args.mode:
#     cls_classifier.load_state_dict(
#         torch.load(
#             args.cls_classifier, map_location='cpu'),
#             strict=True
#     )


# def main():
#   if 'preprocess' in args.mode:
#     loss_fn = torch.nn.CrossEntropyLoss()
#     classifier.eval()
#     cls_classifier.eval()
#     for data in tqdm(val_loader_tgt):
#         log_scale = nn.Parameter(torch.zeros((len(data[0]), 1, 1)).to(device))
#         mean = nn.Parameter(torch.zeros((len(data[0]), 3, 1)).to(device))
#         optim = torch.optim.Adam([log_scale, mean], lr=1e-3)
#         for iters in range(1000):
#             pcs = data[0].to(device).permute(0,2,1)
#             pcs = pcs * log_scale.exp() + mean
#             if args.use_ori:
#                 pcs_t = pcs
#                 t_int = pcs.new_zeros((pcs.size(0), 1))
#             else:
#                 t_int = torch.randint(
#                     0, model.T, # ~= diffusion_steps * 0.4
#                     size=(pcs.size(0), 1), device=device).float()
#                 t = t_int / model.T

#                 gamma_t = model.inflate_batch_array(model.gamma(t), pcs.permute(0,2,1))
#                 alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
#                 sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

#                 node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
#                 eps = model.sample_combined_position_feature_noise(
#                     n_samples=pcs.size(0), n_nodes=pcs.size(2), node_mask=node_mask,
#                     device=device
#                 ).permute(0,2,1)

#                 pcs_t = alpha_t * pcs + sigma_t * eps

#             logits = classifier(pcs_t, ts=t_int.flatten())['domain_cls']

#             loss = loss_fn(logits, logits.new_zeros(len(logits)).long())
#             print(logits.argmax(dim=-1))
#             print(loss)
#             loss.backward()
#             optim.step()
#         print("mean", mean.squeeze())
#         print("scale", scale.squeeze())

#         ori_logits = cls_classifier(data[0].to(device).permute(0,2,1),
#                 ts=torch.zeros_like(t_int).flatten())['cls']
#         trans_logits = cls_classifier(
#                 log_scale.exp() * data[0].to(device).permute(0,2,1) + mean,
#                 ts=torch.zeros_like(t_int).flatten())['cls']
#         ori_pred = ori_logits.argmax(dim=1)
#         ori_prob = ori_logits.softmax(dim=-1).max(dim=1).values
#         trans_pred = trans_logits.argmax(dim=1)
#         trans_prob = trans_logits.softmax(dim=-1).max(dim=1).values

#         labels = [idx_to_label[int(d)] for d in data[1]]
#         ori_pred = [idx_to_label[int(d)] for d in ori_pred]
#         trans_pred = [idx_to_label[int(d)] for d in trans_pred]

#         rgbs = get_color(data[0].cpu().numpy())
#         rgbs_trans = get_color((data[0].to(device) * log_scale.exp() +
#             mean.permute(0,2,1)).detach().cpu().numpy())
#         for b in range(len(pcs)):
#             obj3d = wandb.Object3D({
#                 "type": "lidar/beta",
#                 "points": np.concatenate((data[0][b].cpu().numpy().reshape(-1, 3),
#                     rgbs[b]), axis=1),
#                 "boxes": np.array(
#                     [
#                         {
#                             "corners":
#                             (np.array([
#                                 [-1, -1, -1],
#                                 [-1, 1, -1],
#                                 [-1, -1, 1],
#                                 [1, -1, -1],
#                                 [1, 1, -1],
#                                 [-1, 1, 1],
#                                 [1, -1, 1],
#                                 [1, 1, 1]
#                             ])*2).tolist(),
#                             "label": f'{labels[b]} {ori_pred[b]} {ori_prob[b].item()}',
#                             "color": [123, 321, 111], # ???
#                         }
#                     ]
#                 ),
#             })
#             wandb.log({f'ori': obj3d}, step=b, commit=False)

#             obj3d = wandb.Object3D({
#                 "type": "lidar/beta",
#                 "points": np.concatenate(((log_scale.exp() * data[0].to(device) +
#                     mean.permute(0,2,1))[b].detach().cpu().numpy().reshape(-1, 3),
#                     rgbs_trans[b]), axis=1),
#                 "boxes": np.array(
#                     [
#                         {
#                             "corners":
#                             (np.array([
#                                 [-1, -1, -1],
#                                 [-1, 1, -1],
#                                 [-1, -1, 1],
#                                 [1, -1, -1],
#                                 [1, 1, -1],
#                                 [-1, 1, 1],
#                                 [1, -1, 1],
#                                 [1, 1, 1]
#                                 ])*2).tolist(),
#                             "label": f'{labels[b]} {trans_pred[b]} {trans_prob[b].item()}, {mean[b].squeeze()} {log_scale.exp()[b].squeeze()}',
#                             "color": [123, 321, 111], # ???
#                         }
#                     ]
#                 ),
#             })
#             wandb.log({f'trans': obj3d}, step=b, commit=False)
#         break

#   if 'eval' in args.mode:
#     classifier.eval()
#     n_correct_src = 0
#     n_correct_tgt = 0
#     n_total_src = 0
#     n_total_tgt = 0
#     for val_src in tqdm(val_loader_src):
#         if time_cond:
#             pcs = val_src[0].to(device).permute(0,2,1)
#             t_int = pcs.new_ones((pcs.size(0), 1)) * int(model.T * args.t)
#             #t_int = torch.randint(
#             #    0, model.T, # ~= diffusion_steps * 0.4
#             #    size=(pcs.size(0), 1), device=device).float()
#             t = t_int / model.T

#             gamma_t = model.inflate_batch_array(model.gamma(t),
#                     pcs.permute(0,2,1))
#             alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
#             sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

#             node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
#             eps = model.sample_combined_position_feature_noise(
#                 n_samples=pcs.size(0), n_nodes=pcs.size(2), node_mask=node_mask,
#                 device=device
#             ).permute(0,2,1)

#             pcs_t = alpha_t * pcs + sigma_t * eps
#             logits_src = classifier(pcs_t, ts=t_int.flatten())
#         else:
#             logits_src = classifier(val_src[0].to(device).permute(0,2,1))
#         #print(logits_src['domain_cls'].argmax(dim=1), "source")
#         n_correct_src += (logits_src['domain_cls'].argmax(dim=1) ==
#                 0).float().sum().item()
#         n_total_src += len(logits_src['domain_cls'])

#     for val_tgt in tqdm(val_loader_tgt):
#         if time_cond:
#             pcs = val_tgt[0].to(device).permute(0,2,1)
#             t_int = pcs.new_ones((pcs.size(0), 1)) * int(model.T * args.t)
#             #t_int = torch.randint(
#             #    0, 200, # ~= diffusion_steps * 0.4
#             #    size=(pcs.size(0), 1), device=device).float()
#             t = t_int / model.T
#             gamma_t = model.inflate_batch_array(model.gamma(t),
#                     pcs.permute(0,2,1))
#             alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
#             sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

#             node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
#             eps = model.sample_combined_position_feature_noise(
#                     n_samples=pcs.size(0), n_nodes=pcs.size(2), node_mask=node_mask,
#                 device=device
#             ).permute(0,2,1)

#             pcs_t = alpha_t * pcs + sigma_t * eps
#             pcs_t = pcs
#             t_int = torch.zeros_like(t_int)
#             logits_tgt = classifier(pcs_t, ts=t_int.flatten())
#         else:
#             logits_tgt = classifier(val_tgt[0].to(device).permute(0,2,1))
#         #print(logits_tgt['domain_cls'].argmax(dim=1), "target")
#         n_correct_tgt += (logits_tgt['domain_cls'].argmax(dim=1) ==
#                 1).float().sum().item()
#         n_total_tgt += len(logits_tgt['domain_cls'])
#     print("####### source acc", n_correct_src / n_total_src)
#     print('####### target acc', n_correct_tgt / n_total_tgt)


#   if 'train' in args.mode:
#     tgt_data = 'scan'
#     use_ori = args.use_ori #False
#     torch.save(args,
#             f"outputs/domain_classifier_DGCNN_shape_{tgt_data}_{random_remove}{p_keep}_ori{use_ori}_fullt_fcnorm{args.fc_norm}.args.pt")
#     loss_fn = torch.nn.CrossEntropyLoss()
#     best_val = 0
#     for epoch in range(args.n_epochs):
#         # Train
#         train_count = 0
#         train_correct = 0
#         for i, data_src in enumerate(tqdm(train_loader_src)):
#             classifier.train()
#             try:
#                 data_tgt = next(train_loader_tgt_iter)
#             except:
#                 train_loader_tgt_iter = iter(train_loader_tgt)
#                 data_tgt = next(train_loader_tgt_iter)
#             #data = torch.cat((
#             #    data_src[0].to(device),
#             #    data_tgt[0].to(device)), dim=0)
#             #data = data.permute(0,2,1)
#             data = torch.stack((
#                 data_src[0 if not random_remove else 4].to(device).permute(0, 2, 1),
#                 data_tgt[0].to(device).permute(0, 2, 1)), dim=1)
#             # batchsize x 2 x 3 x 1024
#             data = data.view(-1, data.shape[-2], data.shape[-1])
#             # batch size*2 x 3 x 1024
#             if time_cond:
#                 pcs = data
#                 t_int = torch.randint(
#                     0, int(model.T), # ~= diffusion_steps * 0.4
#                     size=(pcs.size(0), 1), device=device).float()
#                 if use_ori:
#                     t_int = torch.zeros_like(t_int)
#                 t = t_int / model.T

#                 gamma_t = model.inflate_batch_array(model.gamma(t),
#                         pcs.permute(0,2,1))
#                 alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
#                 sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

#                 node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
#                 eps = model.sample_combined_position_feature_noise(
#                     n_samples=pcs.size(0), n_nodes=pcs.size(2),
#                     node_mask=node_mask,
#                     device=device
#                 ).permute(0,2,1)
#                 if use_ori:
#                     pcs_t = pcs
#                 else:
#                     pcs_t = alpha_t * pcs + sigma_t * eps
#                 # TODO: DGCNN에 t conditioning 하기
#                 logits = classifier(
#                     pcs_t, ts=t_int.flatten()
#                     ) #, activate_DefRec=False)
#             else:
#                 # bn 때문에 한번에 pass 해주도록 수정함
#                 logits = classifier(
#                     data
#                     ) #, activate_DefRec=False)
#             logits = logits['domain_cls']
#             labels = torch.stack((logits.new_zeros((len(logits)//2,)),
#                 logits.new_ones((len(logits)//2,))),dim=1).long()
#             labels = labels.flatten()
#             train_correct += (logits.argmax(dim=1) == labels).float().sum().item()
#             train_count += len(logits)

#             loss = loss_fn(logits, labels)
#             print("loss", loss.item())
#             loss.backward()
#             optim.step()
#             optim.zero_grad()
#             print(f"Epoch {epoch} Train acc {train_correct/train_count}")

#             # Val : accuracy check
#         with torch.no_grad():
#             if (epoch + 1) % 1 == 0:
#                 classifier.eval()
#                 n_correct_src = 0
#                 n_correct_tgt = 0
#                 n_total_src = 0
#                 n_total_tgt = 0
#                 for val_src in val_loader_src:
#                     if time_cond:
#                         pcs = val_src[4].to(device).permute(0,2,1)
#                         t_int = torch.randint(
#                             0, model.T, # ~= diffusion_steps * 0.4
#                             size=(pcs.size(0), 1), device=device).float()
#                         t = t_int / model.T
#                         gamma_t = model.inflate_batch_array(model.gamma(t),
#                                 pcs.permute(0,2,1))
#                         alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
#                         sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))
#                         node_mask = pcs.new_ones(pcs.shape[0], pcs.shape[2]).unsqueeze(-1)
#                         eps = model.sample_combined_position_feature_noise(
#                             n_samples=pcs.size(0), n_nodes=pcs.size(2),
#                             node_mask=node_mask,
#                             device=device
#                         ).permute(0,2,1)

#                         pcs_t = alpha_t * pcs + sigma_t * eps
#                         pcs_t = pcs
#                         t_int = torch.zeros_like(t_int)
#                         logits_src = classifier(pcs_t, ts=t_int.flatten())
#                     else:
#                         logits_src = classifier(val_src[4].to(device).permute(0,2,1))
#                     #print(logits_src['domain_cls'].argmax(dim=1), "source")
#                     n_correct_src += (logits_src['domain_cls'].argmax(dim=1) ==
#                             0).float().sum().item()
#                     n_total_src += len(logits_src['domain_cls'])

#                 for val_tgt in val_loader_tgt:
#                     if time_cond:
#                         pcs = val_tgt[0].to(device).permute(0,2,1)

#                         t_int = torch.randint(
#                             0, model.T, # ~= diffusion_steps * 0.4
#                             #0, 200, # ~= diffusion_steps * 0.4
#                             size=(pcs.size(0), 1), device=device).float()
#                         t = t_int / model.T

#                         gamma_t = model.inflate_batch_array(model.gamma(t),
#                                 pcs.permute(0,2,1))
#                         alpha_t = model.alpha(gamma_t, pcs.permute(0,2,1))
#                         sigma_t = model.sigma(gamma_t, pcs.permute(0,2,1))

#                         eps = model.sample_combined_position_feature_noise(
#                             n_samples=pcs.size(0), n_nodes=pcs.size(2),
#                             node_mask=node_mask,
#                             device=device
#                         ).permute(0,2,1)

#                         pcs_t = alpha_t * pcs + sigma_t * eps
#                         pcs_t = pcs
#                         t_int = torch.zeros_like(t_int)
#                         logits_tgt = classifier(pcs_t, ts=t_int.flatten())
#                     else:
#                         logits_tgt = classifier(val_tgt[0].to(device).permute(0,2,1))
#                     #print(logits_tgt['domain_cls'].argmax(dim=1), "target")
#                     n_correct_tgt += (logits_tgt['domain_cls'].argmax(dim=1) ==
#                             1).float().sum().item()
#                     n_total_tgt += len(logits_tgt['domain_cls'])
#                 print("epoch", epoch)
#                 print("source acc", n_correct_src / n_total_src)
#                 print('target acc', n_correct_tgt / n_total_tgt)

#                 val_result = n_correct_src / n_total_src + n_correct_tgt / n_total_tgt

#                 if val_result > best_val:
#                     best_val = val_result
#                     torch.save(classifier.state_dict(),
#                             f"outputs/domain_classifier_DGCNN_shape_{tgt_data}_{random_remove}{p_keep}_ori{use_ori}_fullt_fcnorm{args.fc_norm}.pt")
#                 torch.save(classifier.state_dict(),
#                         f"outputs/domain_classifier_DGCNN_shape_{tgt_data}_{random_remove}{p_keep}_ori{use_ori}_fullt_fcnorm{args.fc_norm}_last.pt")



# if __name__ == "__main__":
#     main()