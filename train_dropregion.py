import numpy as np
import random
import torch
import json
import torch.nn as nn
import torch.optim as optim
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
from data.dataloader_Norm import ScanNet, ModelNet, ShapeNet, label_to_idx, NUM_POINTS
from Models_Norm import PointNet, DGCNN
from utils_GAST import pc_utils_Norm, loss, log
from utils import defcls_input, region_mean
from tqdm import tqdm
import wandb
import utils

NWORKERS = 4
MAX_LOSS = 9 * (10 ** 9)


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--wandb_usr', type=str, default='mazing')
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb') # TODO
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--exp_name', type=str, default='drop_region_pred', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./outputs', help='log folder path')
parser.add_argument('--dataroot', type=str, default='data', metavar='N', help='data path')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=200, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--cls_weight', type=float, default=0.5, help='weight of the classification loss')
parser.add_argument('--grl_weight', type=float, default=0.5, help='weight of the GRL loss')
parser.add_argument('--spl_weight', type=float, default=0.5, help='weight of the SPL loss')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--DefCls_weight', type=float, default=0.5, help='weight of the DefCls loss')
parser.add_argument('--RotCls_weight', type=float, default=0.2, help='weight of the RotCls loss')
parser.add_argument('--NormReg_weight', type=float, default=0.5, help='weight of the NormReg loss')
parser.add_argument('--Decoder_weight', type=float, default=2.0, help='weight of the Decoder loss')
parser.add_argument('--output_pts', type=int, default=512, help='number of decoder points')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--gamma', type=float, default=0.1, help='threshold for pseudo label')
parser.add_argument('--pred_stat', action='store_true', default=True)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--drop_rate', default=0.5, type=float)
parser.add_argument('--rng', default=4, type=float)
parser.add_argument('--nregions', default=3, type=int)
parser.add_argument('--resume',
        default='outputs/drop_region_pred_lr1e-4/model.ptdgcnn', type=str)

args = parser.parse_args()

# ==================
# init
# ==================
io = log.IOStream(args)
io.cprint(str(args))

# =================
# wandb
# =================
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
# Wandb config
if args.no_wandb or args.mode == 'train':
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project':
        'drop_region', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

random.seed(1)
# np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
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


src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

src_trainset = data_func[src_dataset](io, args.dataroot, 'train',
        jitter=False, scale=1, scale_mode='unit_std')
src_valset = data_func[src_dataset](io, args.dataroot, 'val',
        jitter=False, scale=1, scale_mode='unit_std')
src_testset = data_func[src_dataset](io, args.dataroot, 'test',
        jitter=False, scale=1, scale_mode='unit_std')
#trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train',
#        jitter=False, scale=1, scale_mode='unit_std')
#trgt_valset = data_func[trgt_dataset](io, args.dataroot, 'val'
#        jitter=False, scale=1, scale_mode='unit_std')
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test',
        jitter=False, scale=1, scale_mode='unit_std')

# Creating data indices for training and validation splits:
src_train_sampler, src_valid_sampler = None, None #split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = None, None #split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                              sampler=src_train_sampler, drop_last=True,
                              shuffle=True)
src_val_loader = DataLoader(src_valset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                            sampler=src_valid_sampler, shuffle=True)
src_test_loader = DataLoader(src_testset, num_workers=NWORKERS,
        batch_size=args.test_batch_size, shuffle=True)
#trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
#                               sampler=trgt_train_sampler, drop_last=True)
#trgt_val_loader = DataLoader(trgt_valset, num_workers=NWORKERS, batch_size=args.test_batch_size,
#                             sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS,
        batch_size=args.test_batch_size, shuffle=True)

# ==================
if args.model == 'pointnet':
    model = PointNet(args)
elif args.model == 'dgcnn':
    model = DGCNN(args)
else:
    raise Exception("Not implemented")

model = model.to(device)

if args.mode == 'test':
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)



# ==================
# Optimizer
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(opt, args.epochs - 10)
criterion_bce = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
criterion_ls = loss.LabelSmoothingCrossEntropy()
criterion_elem = nn.CrossEntropyLoss(reduction='none')  # return the each sample CE over the batch
# lookup table of regions means
#lookup = torch.Tensor(pc_utils_Norm.region_mean(args.num_regions)).to(device)
lookup = region_mean(num_regions=args.num_regions, rng=args.rng)


# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {'loss_stat': 0.0, 'loss_dropregion':0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data in test_loader:
            data, labels = data[0].to(device), data[1].to(device).squeeze()
            data, mask, drop_region, mean, std = \
                defcls_input(data, NREGIONS=args.nregions, rng=args.rng,
                        drop_rate=args.drop_rate)
            batch_size = data.size()[0]
            logits = model(data.permute(0,2,1), activate_DefRec=False,
                    mask=mask.view(batch_size, -1).bool())
            pred_dropregion = logits['def_cls']
            pred_stat = logits['stat']

            loss_dropregion = criterion_bce(pred_dropregion, drop_region.float())
            pred_stat = torch.cat((pred_stat[:, :3], pred_stat[:, 3:].exp()), dim=1)
            true_stat = torch.cat((mean.squeeze(1), std.squeeze(1)), dim=-1)
            loss_stat = (pred_stat - true_stat).pow(2).sum(-1).mean()
            #true_stat = torch.cat((mean.squeeze(1), torch.log(std).squeeze(1)), dim=-1)
            #loss_stat = (pred_stat - true_stat).pow(2).sum(-1).mean()

            print_losses['loss_stat'] += loss_stat.item() * batch_size
            print_losses['loss_dropregion'] += loss_dropregion.item() * batch_size
            # evaluation metrics
            labels = drop_region.long()
            preds = (pred_dropregion.sigmoid() > 0.5).long()
            #preds = logits["cls"].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = (test_true == test_pred).mean()
    print("test acc", test_acc)
    print(print_losses)

    #test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    #conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['loss_dropregion'] #print_losses['cls'], #conf_mat

def np_softmax(x):

    max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

gray_color = np.array([[255, 255, 255]])
def get_color(coords, corners=np.array([
                                    [-1, -1, -1],
                                    [-1, 1, -1],
                                    [-1, -1, 1],
                                    [1, -1, -1],
                                    [1, 1, -1],
                                    [-1, 1, 1],
                                    [1, -1, 1],
                                    [1, 1, 1]
                                ]) * 4,
    ):
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

    weight = np_softmax(-dist)[:, :, :, None] #batch x NUM_POINTS x n_corners x 1
    rgb = (weight * colors).sum(2).astype(int) # NUM_POINTS x 3
    return rgb


def visualize(test_loader, model=None, set_type="Target", partition="Val", epoch=0):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {'loss_stat': 0.0, 'loss_dropregion':0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        for data in test_loader:
            data, labels = data[0].to(device), data[1].to(device).squeeze()
            batch_size = len(data)
            data_ori = data.clone()
            data, mask, drop_region, mean, std = \
                defcls_input(data, NREGIONS=args.nregions, rng=args.rng, drop_rate=args.drop_rate)
            logits = model(data.permute(0,2,1), activate_DefRec=False,
                    mask=mask.view(batch_size, -1).bool())
            pred_dropregion = logits['def_cls']
            pred_stat = logits['stat']

            labels = drop_region.long()
            preds = (pred_dropregion.sigmoid() > 0.5).long()
            print("# ---- true ----")
            print(labels)
            print("# ---- pred ----")
            print(preds)

            print("------ mean ------")
            print(mean.squeeze())
            print("------ mean pred ------")
            print(pred_stat[:, :3].squeeze())

            print("------ std ------")
            print(std.squeeze())
            print("------ std pred ------")
            print(pred_stat[:, 3:].exp().squeeze())

            ori_rgb = get_color(data_ori.cpu().numpy())
            data_rgb = get_color(data.cpu().numpy())
            #print(data.shape, pred_stat[:, 3:].unsqueeze(1).shape,
            #        pred_stat[:, :3].unsqueeze(1).shape, mask.shape)
            #input()
            data_moved = (data*pred_stat[:, 3:].unsqueeze(1).exp() + pred_stat[:,
                :3].unsqueeze(1)) * mask
            moved_rgb = get_color(data_moved.cpu().numpy())
            for ind in range(len(data)):

                predregion_ind = preds[ind].nonzero().view(-1).cpu().numpy()
                if len(predregion_ind):
                    region_mean = lookup[predregion_ind]
                    new_points = np.random.normal(region_mean, scale=1/3,
                            size=(30,*region_mean.shape)).reshape(-1, 3)
                    new_gray = np.tile(gray_color, (len(new_points), 1))
                    rgb_new_points = get_color(new_points[None, :, :])[0]


                obj3d = wandb.Object3D({
                    "type": "lidar/beta",
                    "points":
                    np.concatenate((
                    np.concatenate((data_ori[ind].cpu().numpy().reshape(-1, 3),
                        np.tile(gray_color, [1024, 1])),
                        axis=1)[(~mask.bool())[ind].squeeze().cpu().numpy()],
                    np.concatenate((data_moved[ind].cpu().numpy(),
                        moved_rgb[ind]), axis=1),
                    ), axis=0),
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
                                ]) * 4).tolist(),
                                "label": f'{mean.squeeze()[ind].cpu().numpy()} {pred_stat[ind, :3].cpu().numpy()} / {std[ind].squeeze().item()}  {pred_stat[ind, 3:].exp().item()}',
                                "color": [123, 321, 111], # ???
                            }
                        ]
                    ),
                })
                wandb.log({f'data': obj3d}, step=ind, commit=False)
                if len(predregion_ind):
                    obj3d = wandb.Object3D({
                        "type": "lidar/beta",
                        "points":
                        np.concatenate((
                            np.concatenate((new_points.reshape(-1, 3),
                        #rgb_new_points
                        new_gray
                        ), axis=1),
                        np.concatenate((data_moved[ind].cpu().numpy(),
                            moved_rgb[ind]), axis=1),
                        ), axis=0),
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
                                    ]) * 4).tolist(),
                                    "label": f'{mean.squeeze()[ind].cpu().numpy()} {pred_stat[ind, :3].cpu().numpy()} / {std[ind].squeeze().item()}  {pred_stat[ind, 3:].exp().item()}',
                                    "color": [123, 321, 111], # ???
                                }
                            ]
                        ),
                    })
                    wandb.log({f'with noise fill': obj3d}, step=ind, commit=False)

                obj3d = wandb.Object3D({
                    "type": "lidar/beta",
                    "points":
                    np.concatenate((data_moved[ind].cpu().numpy(),
                        moved_rgb[ind]), axis=1),
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
                                ]) * 4).tolist(),
                                "label": f'{mean.squeeze()[ind].cpu().numpy()} {pred_stat[ind, :3].cpu().numpy()} / {std[ind].squeeze().item()}  {pred_stat[ind, 3:].exp().item()}',
                                "color": [123, 321, 111], # ???
                            }
                        ]
                    ),
                })
                wandb.log({f'dropped': obj3d}, step=ind, commit=False)
            break
    return


def visualize_target(test_loader, model=None, set_type="Target", partition="Val", epoch=0):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {'loss_stat': 0.0, 'loss_dropregion':0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        for data in test_loader:
            num_points = data[3]
            mask = torch.zeros((len(data[0]), 1024, 1)).to(device)
            mask[num_points.view(-1, 1) > torch.arange(1024)] = 1
            data = data[4].to(device)
            #data, labels = data[4].to(device), data[1].to(device).squeeze()
            batch_size = len(data)
            data_ori = data.clone()
            #data, mask, drop_region, mean, std = \
            #    defcls_input(data, NREGIONS=3, rng=4, drop_rate=args.drop_rate)
            logits = model(data.permute(0,2,1).float(), activate_DefRec=False,
                    mask=mask.view(batch_size, -1).bool())
            pred_dropregion = logits['def_cls']
            preds = (pred_dropregion.sigmoid() > 0.5).long()
            pred_stat = logits['stat']

            data_moved = (data*pred_stat[:, 3:].unsqueeze(1).exp() + pred_stat[:,
                :3].unsqueeze(1)) * mask
            moved_rgb = get_color(data_moved.cpu().numpy())
            for ind in range(len(data)):
                predregion_ind = preds[ind].nonzero().view(-1).cpu().numpy()
                if len(predregion_ind):
                    region_mean = lookup[predregion_ind]
                    new_points = np.random.normal(region_mean, scale=1/3,
                            size=(30,*region_mean.shape)).reshape(-1, 3)
                    new_gray = np.tile(gray_color, (len(new_points), 1))
                    rgb_new_points = get_color(new_points[None, :, :])[0]

                obj3d = wandb.Object3D({
                    "type": "lidar/beta",
                    "points":
                    np.concatenate((
                        np.concatenate((data_moved[ind].cpu().numpy(),
                            moved_rgb[ind]), axis=1),
                        np.concatenate((new_points, new_gray), axis=1),
                    ), axis=0) if len(predregion_ind) else
                        np.concatenate((data_moved[ind].cpu().numpy(),
                            moved_rgb[ind]), axis=1),
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
                                ]) * 4).tolist(),
                                "label": f'{pred_stat[ind, :3].cpu().numpy()} / {pred_stat[ind, 3:].exp().item()}',
                                "color": [123, 321, 111], # ???
                            }
                        ]
                    ),
                })
                wandb.log({f'target': obj3d}) #, step=ind, commit=False)
            break
    return


# ==================
# Utils
# ==================
def generate_trgt_pseudo_label(trgt_data, logits, threshold):
    batch_size = trgt_data.size(0)
    pseudo_label = torch.zeros(batch_size, 10).long()  # one-hot label
    sfm = nn.Softmax(dim=1)
    cls_conf = sfm(logits['cls'])
    mask = torch.max(cls_conf, 1)  # 2 * b
    for i in range(batch_size):
        index = mask[1][i]
        if mask[0][i] > threshold:
            pseudo_label[i][index] = 1

    return pseudo_label


# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
#best_model = io.save_model(model)
src_val_acc_list = []
src_val_loss_list = []
trgt_val_acc_list = []
trgt_val_loss_list = []

#with torch.autograd.set_detect_anomaly(True):
if args.mode == 'train':
  for epoch in range(args.epochs):
    print("Epoch:", epoch)
    model.train()
    len_dataloader = len(src_train_loader)#, len(trgt_train_loader))

    # init data structures for saving epoch stats
    cls_type = 'cls' #'mixup' if args.apply_PCM else 'cls'
    src_print_losses = {} #'total': 0.0, cls_type: 0.0}
    src_print_losses['loss_dropregion'] = 0.0
    src_print_losses['loss_stat'] = 0.0
    src_count = trgt_count = 0.0

    batch_idx = 1
    for data in tqdm(src_train_loader):
        data = data[0].to(device)
        opt.zero_grad()
        #### source data ####
        data, mask, drop_region, mean, std = \
            defcls_input(data, NREGIONS=args.nregions, rng=args.rng, drop_rate=args.drop_rate)
        batch_size = data.size()[0]
        logits = model(data.permute(0,2,1), activate_DefRec=False,
                mask=mask.view(batch_size, -1).bool())
        pred_dropregion = logits['def_cls']
        pred_stat = logits['stat']

        loss_dropregion = criterion_bce(pred_dropregion, drop_region.float())
        #true_stat = torch.cat((mean.squeeze(1), torch.log(std).squeeze(1)), dim=-1)
        pred_stat = torch.cat((pred_stat[:, :3], pred_stat[:, 3:].exp()), dim=1)
        true_stat = torch.cat((mean.squeeze(1), std.squeeze(1)), dim=-1)
        loss_stat = (pred_stat - true_stat).pow(2).sum(-1).mean()

        src_print_losses['loss_stat'] += loss_stat.item() * batch_size
        src_print_losses['loss_dropregion'] += loss_dropregion.item() * batch_size
        (loss_dropregion + loss_stat).backward()
        src_count += batch_size
        opt.step()
        batch_idx += 1

    scheduler.step()

    # print progress
    src_print_losses = {k: v * 1.0 / src_count for (k, v) in src_print_losses.items()}
    src_acc = io.print_progress("Source", "Trn", epoch, src_print_losses)

    # ===================
    # Validation
    # ===================
    src_val_acc, src_val_loss = test(src_val_loader, model, "Source", "Val", epoch)
    #src_val_acc, src_val_loss, src_conf_mat = test(src_val_loader, model, "Source", "Val", epoch)
    #trgt_val_acc, trgt_val_loss, trgt_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch)
    src_val_acc_list.append(src_val_acc)
    src_val_loss_list.append(src_val_loss)
    #trgt_val_acc_list.append(trgt_val_acc)
    #trgt_val_loss_list.append(trgt_val_loss)

    # save model according to best source model (since we don't have target labels)
    if src_val_acc > src_best_val_acc:
        src_best_val_acc = src_val_acc
        src_best_val_loss = src_val_loss
        #trgt_best_val_acc = trgt_val_acc
        #trgt_best_val_loss = trgt_val_loss
        best_val_epoch = epoch
        #best_epoch_conf_mat = trgt_conf_mat
        best_model = io.save_model(model)

    # with open('convergence.json', 'w') as f:
    #    json.dump((src_val_acc_list, src_val_loss_list, trgt_val_acc_list, trgt_val_loss_list), f)

  io.cprint("Best model was found at epoch %d, source validation accuracy: %.4f, source validation loss: %.4f,"
          "target validation accuracy: %.4f, target validation loss: %.4f"
          % (best_val_epoch, src_best_val_acc, src_best_val_loss, trgt_best_val_acc, trgt_best_val_loss))
  io.cprint("Best validtion model confusion matrix:")
  model = best_model

# ===================
# Test
# ===================
if not args.no_wandb:
    visualize(src_test_loader, model, "Source", "Test", 0)
    visualize_target(trgt_test_loader, model, "Target", "Test", 0)
src_test_acc, src_test_loss = test(src_test_loader, model,
        "Source", "Test", 0)
io.cprint("Source test accuracy: %.4f, source test loss: %.4f" % (src_test_acc,
    src_test_loss))
