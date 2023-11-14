import math
from typing import Any, Dict, List, Iterable, Sequence, Tuple, Callable, Union

import torch
import torch.nn as nn


"""
Adapted from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L124
Adapted from: https://github.com/openai/point-e/blob/fc8a607c08a3ea804cc82bf1ef8628f88a3a5d2f/point_e/models/configs.py#L8
"""


MODEL_CONFIGS = {
    "base40M-imagevec": {
        "cond_drop_prob": 0.1,
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 1024,
        "name": "CLIPImagePointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "token_cond": True,
        "width": 512,
    },
    "base40M-textvec": {
        "cond_drop_prob": 0.1,
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 1024,
        "name": "CLIPImagePointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "token_cond": True,
        "width": 512,
    },
    "base40M-uncond": {
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 3, #6, EDIT
        "layers": 12,
        "n_ctx": 1024,
        "name": "PointDiffusionTransformer",
        "output_channels": 3, #12, # EDIT
        "time_token_cond": True,
        "width": 512,
    },
    "base40M": {
        "cond_drop_prob": 0.1,
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 1024,
        "name": "CLIPImageGridPointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "width": 512,
    },
    "base300M": {
        "cond_drop_prob": 0.1,
        "heads": 16,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 24,
        "n_ctx": 1024,
        "name": "CLIPImageGridPointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "width": 1024,
    },
    "base1B": {
        "cond_drop_prob": 0.1,
        "heads": 32,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 24,
        "n_ctx": 1024,
        "name": "CLIPImageGridPointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "width": 2048,
    },
    "upsample": {
        "channel_biases": [0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        "channel_scales": [2.0, 2.0, 2.0, 0.007843137255, 0.007843137255, 0.007843137255],
        "cond_ctx": 1024,
        "cond_drop_prob": 0.1,
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 3072,
        "name": "CLIPImageGridUpsamplePointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "width": 512,
    },
    "sdf": {
        "decoder_heads": 4,
        "decoder_layers": 4,
        "encoder_heads": 4,
        "encoder_layers": 8,
        "init_scale": 0.25,
        "n_ctx": 4096,
        "name": "CrossAttentionPointCloudSDFModel",
        "width": 256,
    },
}



class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads



class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x



class MLP(nn.Module):
    def __init__(self, *, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))



class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)



class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x



class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.backbone = Transformer(
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width)
        self.input_proj = nn.Linear(input_channels, width)
        self.output_proj = nn.Linear(width, output_channels)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        #assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)



def model_from_config(config: Dict[str, Any]) -> nn.Module:
    config = config.copy()
    name = config.pop("name")
    if name == "PointDiffusionTransformer":
        return PointDiffusionTransformer(**config)
    else:
        raise ValueError(f"unknown model name: {name}")


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)