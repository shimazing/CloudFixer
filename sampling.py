import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, assert_correctly_masked


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow):
    n_samples = 1
    n_nodes = args.n_nodes
    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)
    edge_mask = None
    if args.probabilistic_model == 'diffusion':
        chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, keep_frames=100)
        chain = reverse_tensor(chain) # T ~ 0
        # Repeat last frame to see final sample better.
        chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
        x = chain[-1:, :, 0:3]
        # Prepare entire chain.
        x = chain[:, :, 0:3]
    else:
        raise ValueError
    return x


def sample(args, device, generative_model, nodesxsample=torch.tensor([10]),
           fix_noise=False):
    max_n_nodes = args.n_nodes
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)
    node_mask = torch.ones(batch_size, max_n_nodes).unsqueeze(2).to(device)
    edge_mask = None

    if args.probabilistic_model == 'diffusion':
        x = generative_model.sample(batch_size, max_n_nodes, node_mask,
                edge_mask, fix_noise=fix_noise)
        if not args.no_zero_mean:
            assert_correctly_masked(x, node_mask)
            assert_mean_zero_with_mask(x, node_mask)
    else:
        raise ValueError(args.probabilistic_model)
    return x
