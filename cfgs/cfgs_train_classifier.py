import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="DA on Point Clouds")
    parser.add_argument(
        "--exp_name", type=str, default="classifier", help="Name of the experiment"
    )
    parser.add_argument(
        "--out_path", type=str, default="outputs", help="log folder path"
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
    return args