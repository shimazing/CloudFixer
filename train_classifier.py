import os
import random
import copy
import argparse

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from datasets.dataloader import ModelNet40C, PointDA10, ImbalancedDatasetSampler
from classifier.models import PointNet, DGCNNWrapper
from utils import logging as log

NWORKERS = 4
MAX_LOSS = float('inf')

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description="DA on Point Clouds")
parser.add_argument(
    "--exp_name", type=str, default="classifier", help="Name of the experiment"
)
parser.add_argument(
    "--out_path", type=str, default="./experiments", help="log folder path"
)
parser.add_argument(
    "--dataroot", type=str, default="data", metavar="N", help="data path"
)
parser.add_argument(
    "--src_dataset",
    type=str,
    default="shapenet",
    choices=["modelnet", "shapenet", "scannet"],
)
parser.add_argument(
    "--trgt_dataset",
    type=str,
    default="scannet",
    choices=["modelnet", "shapenet", "scannet"],
)
parser.add_argument(
    "--trgt_dataset2",
    type=str,
    default="modelnet",
    choices=["modelnet", "shapenet", "scannet"],
)
parser.add_argument(
    "--epochs", type=int, default=200, help="number of episode to train"
)
parser.add_argument(
    "--model",
    type=str,
    default="dgcnn",
    choices=["pointnet", "dgcnn"],
    help="Model to use",
)
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument(
    "--gpus",
    type=lambda s: [int(item.strip()) for item in s.split(",")],
    default="0",
    help='comma delimited of gpu ids to use. Use "-1" for cpu usage',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    metavar="batch_size",
    help="Size of train batch per domain",
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=32,
    metavar="batch_size",
    help="Size of test batch per domain",
)
parser.add_argument("--optimizer", type=str, default="ADAM", choices=["ADAM", "SGD"])
parser.add_argument(
    "--cls_weight", type=float, default=0.5, help="weight of the classification loss"
)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--wd", type=float, default=5e-5, help="weight decay")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")

args = parser.parse_args()

# ==================
# init
# ==================
io = log.IOStream(args)
io.cprint(str(args))

random.seed(1)
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint(
        "Using GPUs "
        + str(args.gpus)
        + ","
        + " from "
        + str(torch.cuda.device_count())
        + " devices available"
    )
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint("Using CPU")


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
    io.cprint(
        "Occurrences count of classes in "
        + set_type
        + " "
        + domain
        + " train part: "
        + str(dict(zip(unique, counts)))
    )
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint(
        "Occurrences count of classes in "
        + set_type
        + " "
        + domain
        + " validation part: "
        + str(dict(zip(unique, counts)))
    )
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset

args.dataset = args.src_dataset
args.dataset_dir = os.path.join(args.dataroot, "PointDA_data", args.dataset)
src_trainset = PointDA10(args, partition="train")
src_valset = PointDA10(args, partition="val")
src_testset = PointDA10(args, partition="test")
args.dataset = args.trgt_dataset
args.dataset_dir = os.path.join(args.dataroot, "PointDA_data", args.dataset)
trgt_testset = PointDA10(args, partition="test")
args.dataset = args.trgt_dataset2
args.dataset_dir = os.path.join(args.dataroot, "PointDA_data", args.dataset)
trgt_testset2 = PointDA10(args, partition="test")

# dataloaders for source and target
src_train_loader = DataLoader(
    src_trainset,
    num_workers=NWORKERS,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
src_val_loader = DataLoader(
    src_valset, num_workers=NWORKERS, batch_size=args.test_batch_size
)
src_test_loader = DataLoader(
    src_testset, num_workers=NWORKERS, batch_size=args.test_batch_size
)
trgt_test_loader = DataLoader(
    trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size
)
trgt_test_loader2 = DataLoader(
    trgt_testset2, num_workers=NWORKERS, batch_size=args.test_batch_size
)
label_to_idx = src_trainset.label_to_idx

# ==================
# Init Model
# ==================
if args.model == "pointnet":
    model = PointNet(args)
elif args.model == "dgcnn":
    model = DGCNNWrapper("pointda10", 10)
else:
    raise Exception("Not implemented")

model = model.to(device)

# Handle multi-gpu
if (device.type == "cuda") and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

# ==================
# Optimizer
# ==================
opt = (
    optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )
    if args.optimizer == "SGD"
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
)
scheduler = CosineAnnealingLR(opt, args.epochs)
criterion = nn.CrossEntropyLoss() # return the mean of CE over the batch

# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {"cls": 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data in test_loader:
            data, labels = data[0].to(device), data[1].to(device).squeeze()
            batch_size = data.size()[0]

            logits = model(data)
            loss = criterion(logits, labels)
            print_losses["cls"] += loss.item() * batch_size

            # evaluation metrics
            preds = logits.max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(
        set_type, partition, epoch, print_losses, test_true, test_pred
    )
    conf_mat = metrics.confusion_matrix(
        test_true, test_pred, labels=list(label_to_idx.values())
    ).astype(int)

    return test_acc, print_losses["cls"], conf_mat


# ==================
# Train
# ==================
src_best_val_acc = best_val_epoch = 0
src_best_val_loss = MAX_LOSS
src_val_acc_list = []
src_val_loss_list = []

for epoch in range(args.epochs):
    model.train()

    # init data structures for saving epoch stats
    cls_type = "cls"  #'mixup' if args.apply_PCM else 'cls'
    src_print_losses = {"total": 0.0, cls_type: 0.0}
    src_count = 0.0

    batch_idx = 1
    for data in src_train_loader:
        opt.zero_grad()
        #### source data ####
        src_data, src_label = data[0].to(device), data[1].to(device).squeeze()
        batch_size = src_data.size()[0]
        device = torch.device(
            "cuda:" + str(src_data.get_device()) if args.cuda else "cpu"
        )
        src_logits = model(src_data)
        loss = args.cls_weight * criterion(src_logits, src_label)
        src_print_losses["cls"] += loss.item() * batch_size
        src_print_losses["total"] += loss.item() * batch_size
        loss.backward()

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
    src_val_acc, src_val_loss, src_conf_mat = test(
        src_val_loader, model, "Source", "Val", epoch
    )
    src_val_acc_list.append(src_val_acc)
    src_val_loss_list.append(src_val_loss)

    # save model according to best source model (since we don't have target labels)
    if src_val_acc > src_best_val_acc:
        src_best_val_acc = src_val_acc
        src_best_val_loss = src_val_loss
        best_val_epoch = epoch
        best_model = io.save_model(model)

io.cprint(
    "Best model was found at epoch %d, source validation accuracy: %.4f, source validation loss: %.4f,"
    % (best_val_epoch, src_best_val_acc, src_best_val_loss)
)

# ===================
# Test
# ===================
model = best_model

trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(
    src_test_loader, model, "Source", "Test", 0
)
io.cprint(
    "source test accuracy: %.4f, source test loss: %.4f"
    % (trgt_test_acc, trgt_test_loss)
)
io.cprint(f"{args.src_dataset}")
io.cprint("Test confusion matrix:")
io.cprint("\n" + str(trgt_conf_mat))

trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(
    trgt_test_loader, model, "Target", "Test", 0
)
io.cprint(
    "target test accuracy: %.4f, target test loss: %.4f"
    % (trgt_test_acc, trgt_test_loss)
)
io.cprint(f"{args.trgt_dataset}")
io.cprint("Test confusion matrix:")
io.cprint("\n" + str(trgt_conf_mat))

trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(
    trgt_test_loader2, model, "Target", "Test", 0
)
io.cprint(
    "target2 test accuracy: %.4f, target test loss: %.4f"
    % (trgt_test_acc, trgt_test_loss)
)
io.cprint(f"{args.trgt_dataset2}")
io.cprint("Test confusion matrix:")
io.cprint("\n" + str(trgt_conf_mat))
