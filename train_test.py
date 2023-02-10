import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask,\
    remove_mean
import numpy as np
import visualizer as vis
from sampling import sample_chain, sample
import utils
import losses
import time
import torch


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
        node_mask = None
        edge_mask = None

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        if not args.no_zero_mean:
            x = remove_mean(x)

        h = {}
        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp,
                                                                x, h, node_mask,
                                                                edge_mask)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
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
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            save_and_sample_chain(model_ema, args, device, epoch=epoch,
                                  batch_id=str(i))
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
                                ])*args.scale).tolist(),
                            "label": "Box",
                            "color": [123, 321, 111], # ???
                        }
                    ]
                ),
            })
            wandb.log({'3d_example': obj3d})

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}",
                    wandb=wandb, scale=args.scale)
            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}_ema",
                    wandb=wandb, postfix="_ema",
                    scale=args.scale)
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)

    if (i+1) % args.accum_grad:
        optim.step()
        optim.zero_grad()


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            try:
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
                h = {'categorical': one_hot, 'integer': charges}
                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot, charges], node_mask)
                if not args.no_zero_mean:
                    assert_mean_zero_with_mask(x, node_mask)
            except:
                x = data[0].to(device, dtype)
                node_mask = None
                edge_mask = None
                one_hot = None
                charges = None
                h = {}
                x = remove_mean(x)
            batch_size = x.size(0)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise
            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, x, h,
                                                    node_mask, edge_mask)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, epoch=0, batch_id=''):
    x = sample_chain(args=args, device=device, flow=model)
    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
            x, name='chain', n_nodes=args.n_nodes)


def sample_and_save(model, args, device, n_samples=5, epoch=0, batch_id=''):
    model.eval()
    nodesxsample = torch.tensor([args.n_nodes]*n_samples)
    x = sample(args, device, model, nodesxsample=nodesxsample)
    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/',
             x, name='pointcloud', n_nodes=args.n_nodes)
    model.train()
