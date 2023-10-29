import time
import wandb

import numpy as np
import torch

import utils
import losses
import visualizer as vis


def sample(args, device, generative_model, nodesxsample=torch.tensor([10]),
           fix_noise=False):
    max_n_nodes = args.n_nodes
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)
    node_mask = torch.ones(batch_size, max_n_nodes).unsqueeze(2).to(device)

    if args.probabilistic_model == 'diffusion':
        x = generative_model.sample(batch_size, max_n_nodes, node_mask, fix_noise=fix_noise)
    else:
        raise ValueError(args.probabilistic_model)
    return x


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device,
        dtype, optim,
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
            grad_norm = 0.

        if (i+1) % args.accum_grad == 0:
            optim.step()
            optim.zero_grad()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            sample_and_save(model_ema, args, device, epoch=epoch,
                            batch_id=str(i) + "_ema")
            sample_and_save(model, args, device, epoch=epoch,
                                  batch_id=str(i))
            print(f'Sampling took {time.time() - start:.2f} seconds')

            obj3d = wandb.Object3D({
                "type": "lidar/beta",
                "points": x[0].cpu().numpy().reshape(-1, 3),
                "boxes": np.array(
                    [
                        {
                            "corners": (np.array([
                                [-1, -1, -1],
                                [-1, 1, -1],
                                [-1, -1, 1],
                                [1, -1, -1],
                                [1, 1, -1],
                                [-1, 1, 1],
                                [1, -1, 1],
                                [1, 1, 1]
                                ])*3).tolist(),
                            "label": "Box",
                            "color": [123, 321, 111], # ???
                        }
                    ]
                ),
            })
            wandb.log({'3d_example': obj3d})
            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}",
                    wandb=wandb)
            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}_ema",
                    wandb=wandb, postfix="_ema")
        wandb.log({"Batch NLL": nll.item()}, commit=True)
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    if (i+1) % args.accum_grad:
        optim.step()
        optim.zero_grad()


def test(args, loader, epoch, eval_model, device, dtype, partition='Test'):
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
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def sample_and_save(model, args, device, n_samples=5, epoch=0, batch_id=''):
    model.eval()
    nodesxsample = torch.tensor([args.n_nodes]*n_samples)
    x = sample(args, device, model, nodesxsample=nodesxsample)
    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/',
             x, name='pointcloud', n_nodes=args.n_nodes)
    model.train()