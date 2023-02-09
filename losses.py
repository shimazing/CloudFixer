import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    if node_mask is not None:
        assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, x, h, node_mask, edge_mask):
    bs, n_nodes, n_dims = x.size()
    if args.probabilistic_model == 'diffusion':
        if edge_mask is not None:
            edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = generative_model(x, h, node_mask, edge_mask)
        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z
