import os
import copy
import time
import pickle
import argparse
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader

from cfgs.cfgs_train_dm import parse_arguments
from datasets.dataloader import (
    ModelNet40C,
    PointDA10,
    GraspNet10,
    ImbalancedDatasetSampler,
)
from diffusion import diffusion, utils as flow_utils
from diffusion.build_model import get_model, get_optim
from utils import losses, utils, visualizer as vis


def sample(
    args, device, generative_model, nodesxsample=torch.tensor([10]), fix_noise=False
):
    max_n_nodes = args.n_nodes
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)
    node_mask = torch.ones(batch_size, max_n_nodes).unsqueeze(2).to(device)

    if args.probabilistic_model == "diffusion":
        x = generative_model.sample(
            batch_size, max_n_nodes, node_mask, fix_noise=fix_noise
        )
    else:
        raise ValueError(args.probabilistic_model)
    return x


def train_epoch(
    args,
    loader,
    epoch,
    model,
    model_dp,
    model_ema,
    ema,
    device,
    dtype,
    optim,
    gradnorm_queue,
):
    model_dp.train()
    model.train()
    nll_epoch = []

    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data[0].to(device, dtype)
        # transform batch through flow
        nll = loss = losses.compute_loss_and_nll(args, model_dp, x)
        # standard nll from forward KL
        loss = loss / args.accum_grad
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.0

        if (i + 1) % args.accum_grad == 0:
            optim.step()
            optim.zero_grad()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(
                f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                f"Loss {loss.item():.2f}, "
                f"GradNorm: {grad_norm:.1f}"
            )
        nll_epoch.append(nll.item())

        if (
            (epoch % args.test_epochs == 0)
            and (i % args.visualize_every_batch == 0)
            and not (epoch == 0 and i == 0)
        ):
            start = time.time()
            sample_and_save(
                model_ema, args, device, epoch=epoch, batch_id=str(i) + "_ema"
            )
            sample_and_save(model, args, device, epoch=epoch, batch_id=str(i))
            print(f"Sampling took {time.time() - start:.2f} seconds")

            obj3d = wandb.Object3D(
                {
                    "type": "lidar/beta",
                    "points": x[0].cpu().numpy().reshape(-1, 3),
                    "boxes": np.array(
                        [
                            {
                                "corners": (
                                    np.array(
                                        [
                                            [-1, -1, -1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [1, -1, -1],
                                            [1, 1, -1],
                                            [-1, 1, 1],
                                            [1, -1, 1],
                                            [1, 1, 1],
                                        ]
                                    )
                                    * 3
                                ).tolist(),
                                "label": "Box",
                                "color": [123, 321, 111],  # ???
                            }
                        ]
                    ),
                }
            )
            wandb.log({"3d_example": obj3d})
            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", wandb=wandb)
            vis.visualize(
                f"outputs/{args.exp_name}/epoch_{epoch}_{i}_ema",
                wandb=wandb,
                postfix="_ema",
            )
        wandb.log({"Batch NLL": nll.item()}, commit=True)
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    if (i + 1) % args.accum_grad:
        optim.step()
        optim.zero_grad()


def test(args, loader, epoch, eval_model, device, dtype, partition="Test"):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data[0].to(device, dtype)
            node_mask = None
            batch_size = x.size(0)

            # transform batch through flow
            nll = losses.compute_loss_and_nll(args, eval_model, x)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(
                    f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"NLL: {nll_epoch/n_samples:.2f}"
                )

    return nll_epoch / n_samples


def sample_and_save(model, args, device, n_samples=5, epoch=0, batch_id=""):
    model.eval()
    nodesxsample = torch.tensor([args.n_nodes] * n_samples)
    x = sample(args, device, model, nodesxsample=nodesxsample)
    vis.save_xyz_file(
        f"outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/",
        x,
        name="pointcloud",
        n_nodes=args.n_nodes,
    )
    model.train()


def main(args):
    if args.dataset.startswith("modelnet40"):
        dataset_ = ModelNet40C(args, partition="train")
        dataset_val = ModelNet40C(args, partition="test")
    elif args.dataset in ["modelnet", "shapenet", "scannet"]:
        dataset_ = PointDA10(args=args, partition="train")
        dataset_val = PointDA10(args=args, partition="val")
    elif args.dataset in ["synthetic", "kinect", "realsense"]:
        dataset_ = GraspNet10(args=args, partition="train")
        dataset_val = GraspNet10(args=args, partition="val")
    else:
        raise ValueError("UNDEFINED DATASET")
    train_loader = DataLoader(
        dataset_,
        batch_size=args.batch_size,
        sampler=None if not args.cls_uniform else ImbalancedDatasetSampler(dataset_),
        drop_last=True,
        shuffle=True if not args.cls_uniform else False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32

    if args.resume is not None:
        exp_name = args.exp_name + "_resume"
        start_epoch = args.start_epoch
        resume = args.resume
        wandb_usr = args.wandb_usr
        try:
            with open(os.path.join(args.resume, "args.pickle"), "rb") as f:
                args_ = pickle.load(f)
        except:
            pass

        args.resume = resume
        args.exp_name = exp_name
        args.start_epoch = start_epoch
        args.wandb_usr = wandb_usr
        print(args)

    utils.create_folders(args)

    # Wandb config
    if args.no_wandb:
        mode = "disabled"
    else:
        mode = "online" if args.online else "offline"
    kwargs = {
        "entity": args.wandb_usr,
        "name": args.exp_name,
        "project": "pc_diffusionTTA",
        "config": args,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": mode,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")

    # Create Model
    model = get_model(args, device)
    model = model.to(device)
    optim = get_optim(args, model)

    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    if args.lr_gamma < 1:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, args.lr_gamma)
    if args.resume is not None:
        flow_state_dict = torch.load(
            os.path.join(args.resume, "generative_model_last.npy")
        )
        optim_state_dict = torch.load(os.path.join(args.resume, "optim_last.npy"))
        if args.lr_gamma < 1:
            scheduler_state_dict = torch.load(
                os.path.join(args.resume, "lr_scheduler_last.npy")
            )
            lr_scheduler.load_state_dict(scheduler_state_dict)
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        print("Resume")

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f"Training using {torch.cuda.device_count()} GPUs")
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)
        if args.resume is not None:
            ema_state_dict = torch.load(
                os.path.join(args.resume, "generative_model_ema_last.npy")
            )
            model_ema.load_state_dict(ema_state_dict)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    utils.set_seed(args.random_seed)

    best_nll_val = float("inf")
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.lr_gamma < 1:
            print("LR: ", lr_scheduler.get_last_lr())
        start_time = time.time()
        train_epoch(
            args=args,
            loader=train_loader,
            epoch=epoch,
            model=model,
            model_dp=model_dp,
            model_ema=model_ema,
            ema=ema,
            device=device,
            dtype=dtype,
            gradnorm_queue=gradnorm_queue,
            optim=optim,
        )
        print(f"Epoch took {time.time() - start_time:.1f} seconds.")
        if args.lr_gamma < 1:
            lr_scheduler.step()

        if hasattr(model.dynamics, "report_neighbor_stats"):
            pass
            model.dynamics.report_neighbor_stats()

        if epoch % args.test_epochs == 0 and epoch > 0:
            if isinstance(model, diffusion.DiffusionModel):
                wandb.log(model.log_info(), commit=True)
            nll_val = test(
                args=args,
                loader=val_loader,
                epoch=epoch,
                eval_model=model_ema_dp,
                partition="Val",
                device=device,
                dtype=dtype,
            )

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                if args.save_model:
                    args.current_epoch = epoch + 1
                    if args.lr_gamma < 1:
                        utils.save_model(
                            lr_scheduler, f"{args.output_dir}/%s/lr_scheduler.npy" % args.exp_name
                        )
                    utils.save_model(optim, f"{args.output_dir}/%s/optim.npy" % args.exp_name)
                    utils.save_model(
                        model, f"{args.output_dir}/%s/generative_model.npy" % args.exp_name
                    )
                    if args.ema_decay > 0:
                        utils.save_model(
                            model_ema,
                            f"{args.output_dir}/%s/generative_model_ema.npy" % args.exp_name,
                        )
                    with open(f"{args.output_dir}/%s/args.pickle" % args.exp_name, "wb") as f:
                        pickle.dump(args, f)

                if args.save_model:
                    utils.save_model(
                        optim, f"{args.output_dir}/%s/optim_%d.npy" % (args.exp_name, epoch)
                    )
                    utils.save_model(
                        model,
                        f"{args.output_dir}/%s/generative_model_%d.npy" % (args.exp_name, epoch),
                    )
                    if args.ema_decay > 0:
                        utils.save_model(
                            model_ema,
                            f"{args.output_dir}/%s/generative_model_ema_%d.npy"
                            % (args.exp_name, epoch),
                        )
                    with open(
                        f"{args.output_dir}/%s/args_%d.pickle" % (args.exp_name, epoch), "wb"
                    ) as f:
                        pickle.dump(args, f)

            if args.save_model:
                if args.lr_gamma < 1:
                    utils.save_model(
                        lr_scheduler,
                        f"{args.output_dir}/%s/lr_scheduler_%s.npy" % (args.exp_name, "last"),
                    )
                utils.save_model(
                    optim, f"{args.output_dir}/%s/optim_%s.npy" % (args.exp_name, "last")
                )
                utils.save_model(
                    model,
                    f"{args.output_dir}/%s/generative_model_%s.npy" % (args.exp_name, "last"),
                )
                if args.ema_decay > 0:
                    utils.save_model(
                        model_ema,
                        f"{args.output_dir}/%s/generative_model_ema_%s.npy"
                        % (args.exp_name, "last"),
                    )
                with open(
                    f"{args.output_dir}/%s/args_%s.pickle" % (args.exp_name, "last"), "wb"
                ) as f:
                    pickle.dump(args, f)

            print("Val loss: %.4f" % nll_val)
            print("Best val loss: %.4f" % best_nll_val)
            wandb.log({"Val loss ": nll_val}, commit=True)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
