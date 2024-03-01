""" Vision Transformer (ViT) for Point Cloud Understanding in PyTorch
Hacked together by / Copyright 2020, Ross Wightman
Modified to 3D application by / Copyright 2022@Pix4Point team
"""
import logging
from typing import List
import torch
import torch.nn as nn
from ..layers import create_norm, create_linearblock, create_convblock1d, three_interpolation, \
    furthest_point_sample, random_sample
from ..layers.attention import Block
from .pointnext import FeaturePropogation
from ..build import MODELS, build_model_from_cfg
from torch.cuda.amp import custom_fwd, custom_bwd
import math
import sys

from ..layers import Mlp, DropPath, trunc_normal_, lecun_normal_

use_inv = True

class RevBackProp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, blocks, pos_embed=None, alpha=0., lambd=1., num_cached = 0):
        # print('during rev, alpha, lambda:', alpha, lambd) # TODO: DEBUG

        # hp, number of cached layers

        num_blks = len(blocks)
        buffer_layers = [] 
        if num_blks > 2:
            cached_layer = int(num_blks / (num_cached + 1))
            for i in range(1, num_cached + 1):
                buffer_layers.append(cached_layer * i - 1)

        intermediate = [] 

        for idx, block in enumerate(blocks):
            x = block(x, pos_embed, alpha=alpha, lambd=lambd)

            if idx in buffer_layers:
                intermediate.extend([x.detach()])  # for debug    

        if len(buffer_layers) == 0:
            all_tensors = [x.detach(), pos_embed.detach() if pos_embed is not None else pos_embed]
        else:
            intermediate = [torch.LongTensor(buffer_layers), *intermediate] # for debug
            all_tensors = [x.detach(), pos_embed.detach() if pos_embed is not None else pos_embed, *intermediate] # for debug            

        ctx.save_for_backward(*all_tensors)
        ctx.blocks = blocks
        ctx.lambd = lambd
        ctx.alpha = alpha

        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):  # pragma: no cover

         # retrieve params from ctx for backward
         Y, pos_embed, *intermediate = ctx.saved_tensors     
         blocks = ctx.blocks   
         lambd = ctx.lambd
         alpha = ctx.alpha               
         
         if len(intermediate) != 0:
             buffer_layers = intermediate[0].tolist()
         else:
             buffer_layers = []

         dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)
         Y_1, Y_2 = torch.chunk(Y, 2, dim=-1)

         for idx_i, blk in enumerate(blocks[::-1]):
            idx = len(blocks) - idx_i - 1 

            if idx in buffer_layers:
                Y_intermediate = intermediate[buffer_layers.index(idx) + 1]
                Y_1_inter, Y_2_inter = torch.chunk(Y_intermediate, 2, dim=-1)
                                
                Y_1, Y_2, dY_1, dY_2 = blk.backward_pass(
                    Y_1=Y_1_inter, 
                    Y_2=Y_2_inter,
                    dY_1=dY_1,
                    dY_2=dY_2,
                    pos_embed=pos_embed,
                    alpha=alpha,
                    lambd=lambd,
                    )            
            else:
                Y_1, Y_2, dY_1, dY_2 = blk.backward_pass(
                    Y_1=Y_1, 
                    Y_2=Y_2,
                    dY_1=dY_1,
                    dY_2=dY_2,
                    pos_embed=pos_embed,
                    alpha=alpha,
                    lambd=lambd,
                    )

         dx = torch.cat([dY_1, dY_2], dim=-1)
         del Y_1, Y_2, dY_1, dY_2
         return dx, None, None, None, None, None

def seed_cuda():

    # randomize seeds
    # use cuda generator if available
    if (
        hasattr(torch.cuda, "default_generators")
        and len(torch.cuda.default_generators) > 0
    ):
        # GPU
        device_idx = torch.cuda.current_device()
        seed = torch.cuda.default_generators[device_idx].seed()
    else:
        # CPU
        seed = int(torch.seed() % sys.maxsize)

    return seed

class InvFuncWrapper(nn.Module):
    def __init__(self, Fm, Gm, split_dim=-1):

        super(InvFuncWrapper, self).__init__()
        
        self.Fm = Fm
        self.Gm = Gm

        self.split_dim = split_dim

    def forward(self, x, pos_embed=None, alpha=0., lambd=1.):

        # torch.manual_seed(2022)
        # print(x.shape)
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
        # x1, x2 = x1.contiguous(), x2.contiguous()
        fmd = self.Fm(x2, pos_embed, alpha)
        y1 = lambd * x1 + fmd
        del x1

        # torch.manual_seed(2022)
        gmd = self.Gm(y1, alpha)
        y2 = lambd * x2 + gmd
        del x2

        out = torch.cat([y1, y2], dim=self.split_dim)
        return out

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2, pos_embed=None, alpha=0., lambd=1.):

        """
        equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP

        equations for recompuation of activations:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        """
        # device = Y_1.device
        # print(f'Before Gm: {torch.cuda.memory_allocated(device)}')
        with torch.enable_grad():
            Y_1.requires_grad = True
            # print('backward alpha: ', alpha)
            g_Y_1 = self.Gm(Y_1, alpha)
            # print(f'After Gm: {torch.cuda.memory_allocated(device)}')
            g_Y_1.backward(dY_2, retain_graph=True)
            # print(f'After backward: {torch.cuda.memory_allocated(device)}')
        # print(f'After 1: {torch.cuda.memory_allocated(device)}')

        with torch.no_grad():
            X_2 = (Y_2 - g_Y_1) / lambd
            del g_Y_1
            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None
        # print(f'After 2: {torch.cuda.memory_allocated(device)}\n')
        # print('*************************************************')

        # print(f'Before Fm: {torch.cuda.memory_allocated(device)}')
        with torch.enable_grad():
            X_2.requires_grad = True
            f_X_2 = self.Fm(X_2, pos_embed, alpha)
            # print(f'After Fm: {torch.cuda.memory_allocated(device)}')
            f_X_2.backward(dY_1, retain_graph=True)
            # print(f'After backward: {torch.cuda.memory_allocated(device)}')

        with torch.no_grad():
            X_1 = (Y_1 - f_X_2) / lambd
            del f_X_2, Y_1
            dY_2 = lambd * dY_2 + X_2.grad
            X_2.grad = None
            X_2 = X_2.detach()

            dY_1 = lambd * dY_1
        # print(f'After 4: {torch.cuda.memory_allocated(device)}\n')
        # print('*************************************************')

        return X_1, X_2, dY_1, dY_2


'''
File Description: attention layer for transformer. borrowed from TIMM
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # this is different from what I understand before. 
        # the num_heads here actually works as groups for shared attentions. it partition the channels to different groups.
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple), shape [B, #Heads, N, C]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_inv_F(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., norm_args={'norm': 'ln'}, drop_path=None):
        super().__init__()

        self.drop_path = drop_path
        self.norm1 = create_norm(norm_args, dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.seeds = {}
        self.seeds['droppath'] = seed_cuda()        

    def forward(self, x, pos_embed=None, alpha=0):
        # print('f block seed: ', self.seeds['droppath'])
        # print('inside F alpha: ', alpha)
        torch.manual_seed(self.seeds['droppath'])
        # print('L216: pos_embed is None?: ', pos_embed==None)
        if pos_embed is not None:
            x = x + pos_embed
        skip = x*alpha
        # print('L219:', pos_embed)
        x = skip + self.drop_path(self.attn(self.norm1(x)))
        return x

class Block_inv_G(nn.Module):

    def __init__(self, dim,  mlp_ratio=4., drop=0., act_args={'act': 'gelu'}, norm_args={'norm': 'ln'}, drop_path=None):
        super().__init__()
        
        self.drop_path = drop_path
        self.norm2 = create_norm(norm_args, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # invertable bottleneck layer
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_args=act_args, drop=drop)
        self.seeds = {}
        self.seeds['droppath'] = seed_cuda()

    def forward(self, x, alpha=0):
        torch.manual_seed(self.seeds['droppath'])
        # print('g block seed: ', self.seeds['droppath'])
        # print('inside G alpha: ', alpha)
        skip = alpha*x
        x = skip + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_args={'act': 'gelu'}, norm_args={'norm': 'ln'}):
        super().__init__()

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        Fm = Block_inv_F(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, norm_args=norm_args, drop_path=drop_path)
        Gm = Block_inv_G(dim=dim,  mlp_ratio=mlp_ratio, drop=drop, act_args=act_args, norm_args=norm_args, drop_path=drop_path)

        if use_inv:
            self.inv_block = InvFuncWrapper(Fm=Fm, Gm=Gm, split_dim=-1)
        else:
            self.Fm = Fm
            self.Gm = Gm
            
    def forward(self, x, pos_embed=None, alpha=0., lambd=1.):
        if use_inv:
            x = self.inv_block(x, pos_embed, alpha=alpha, lambd=lambd)
        else:
            x = self.Fm(x, pos_embed, alpha=1)
            x = self.Gm(x, alpha=1)

        return x

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2, pos_embed=None, alpha=0., lambd=1.): 
        return self.inv_block.backward_pass(
                    Y_1=Y_1, 
                    Y_2=Y_2, 
                    dY_1=dY_1, 
                    dY_2=dY_2,
                    pos_embed=pos_embed,
                    alpha=alpha, lambd=lambd)

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 act_args={'act': 'gelu'}, norm_args={'norm': 'ln'}
                 ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                norm_args=norm_args, act_args=act_args
            )
            for i in range(depth)])
        self.depth = depth
        # dilation = depth//3
        # self.out_depth = list(range(depth))[dilation-1::dilation]

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            # x = block(x)

            # Injecting the positional information at each block.
            #  (Dehghani et al., 2018) and (Lan et al.,2019) observe better performance
            #  by further injecting the position information at each block
            # Reference: Learning to Encode Position for Transformer with Continuous Dynamical Model.
            # http://proceedings.mlr.press/v119/liu20n/liu20n.pdf
            x = block(x + pos)
        return x

    def forward_features(self, x, pos, num_outs=None):
        dilation = self.depth // num_outs
        out_depth = list(range(self.depth))[(self.depth - (num_outs-1)*dilation -1) :: dilation]
        
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in out_depth:
                out.append(x)
        return out

@MODELS.register_module()
class InvPointViT(nn.Module):
    """ Point Vision Transformer ++: with early convolutions
    """
    def __init__(self,
                 in_channels=3,
                 embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_args={'NAME': 'PointPatchEmbed', 
                             'num_groups': 256,
                             'group_size': 32,
                             'subsample': 'fps', 
                             'group': 'knn', 
                             'feature_type': 'fj',
                             'norm_args': {'norm': 'in2d'},
                             }, 
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 add_pos_each_block=True,
                 global_feat='cls,max',
                 distill=False, 
                 init_epoch = 0, 
                 start_epoch = 1, 
                 end_epoch = 200, 
                 stop_epoch = -1, 
                 freq=2, 
                 init_lambd = 0.1, 
                 lambd = 1., 
                 alpha = 0.,  
                 option = -1,    
                 freeze_ratio = 0.6,    
                 use_step = False, 
                 num_training_steps_per_epoch=250,
                 use_customized_backprop=True, 
                 num_cached=0,
                 **kwargs
                 ):
        """
        Args:
            in_channels (int): number of input channels. Default: 6. (p + rgb)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        embed_args.in_channels = in_channels
        embed_args.embed_dim = embed_dim
        self.patch_embed = build_model_from_cfg(embed_args)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=None, act_args=act_args),
            nn.Linear(128, self.embed_dim)
        )
        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.add_pos_each_block = add_pos_each_block
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_args=norm_args, act_args=act_args
            )
            for i in range(depth)])
        self.norm = create_norm(norm_args, self.embed_dim)  # Norm layer is extremely important here!
        self.global_feat = global_feat.split(',')
        self.out_channels = len(self.global_feat)*embed_dim
        self.distill_channels = embed_dim
        self.channel_list = self.patch_embed.channel_list
        self.channel_list[-1] = embed_dim

        # distill
        if distill:
            self.dist_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.dist_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.n_tokens = 2
        else:
            self.dist_token = None
            self.n_tokens = 1
        self.initialize_weights()

        self.init_epoch = init_epoch
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.stop_epoch = stop_epoch
        self.freq = freq
        self.alpha = alpha 
        self.init_lambd = init_lambd
        self.lambd = lambd
        self.option = option   
        self.use_step = use_step 
        self.freeze_ratio = freeze_ratio
        self.num_training_steps_per_epoch = num_training_steps_per_epoch 
        self.use_customized_backprop = use_customized_backprop
        self.num_cached = num_cached
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_pos, std=.02)
        if self.dist_token is not None:
            torch.nn.init.normal_(self.dist_token, std=.02)
            torch.nn.init.normal_(self.dist_pos, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'dist_token', 'dist_token'}

    def get_num_layers(self):
        return self.depth

    def get_alpha_lambd_old(self, epoch=-1, step=-1):
        if self.stop_epoch > 0:
            epoch = epoch if epoch < self.stop_epoch else self.stop_epoch
            step = step if epoch < self.stop_epoch else (self.stop_epoch-1) * 2000 / 8

        # print(f'during rev in training: {self.training}, epoch: {epoch}, step: {step}') # TODO: DEBUG
        if self.use_step:
            start_epoch = (self.start_epoch-1) * 2000  / 8
            end_epoch = (self.end_epoch-1) * 2000 / 8
            epoch = step
        else:
            start_epoch = self.start_epoch
            end_epoch = self.end_epoch
         
        if self.option != -1:
            ###################### alpha ######################
            if epoch < self.init_epoch:  # Inference
                alpha = self.alpha
            elif epoch < start_epoch:
                alpha = 1.0
            elif epoch >= start_epoch and epoch < end_epoch:
                if self.option == 6:
                    alpha = 1.
                # Option 0: linear, interleave alpha and lambda
                elif self.option == 0.0:
                    alpha = round(min(max(-round(self.freq/(end_epoch-start_epoch), 12) * math.ceil((epoch-start_epoch-1-self.freq//2) / self.freq) + 1, 0), 1.), 4)
                # self.option 1: linear alpha = (1/(A+1-N))*x + N/(N-A-1), lambd = 1 - alpha + 0.01
                elif self.option == 1:
                    alpha = round(1/(start_epoch+1-end_epoch) * epoch + end_epoch/(end_epoch-start_epoch-1), 2)
                # self.option 2: logarithmic
                elif self.option == 2:
                    base = math.e
                    a = 1/(math.log((start_epoch+1) / end_epoch, base))
                    c = - a * math.log(end_epoch, base)
                    alpha = round(max(min(a *  math.log(epoch, base) + c, 1.), 0), 2)
                # self.option 3: exponential
                elif self.option == 3:
                    base = math.e
                    a = 1/(pow(base, start_epoch+1) - pow(base, end_epoch))
                    c = - a * pow(base, end_epoch)
                    alpha = round(max(min(a * pow(base, epoch) + c, 1.), 0), 2)
                # self.option 4: hand-crafted, increasing
                elif self.option == 4:
                    alpha_list = [1., 1., 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.]
                    alpha = alpha_list[epoch - 1]
                elif self.option == 5:
                    alpha = self.alpha 
            elif epoch >= end_epoch:
                if self.option ==6:
                    alpha = 1.
                else:
                    alpha = self.alpha

            ###################### lambda #####################
            if epoch < self.init_epoch:  # Inference
                lambd = 1.0
            elif epoch < start_epoch:
                lambd = 0.0
            elif epoch >= start_epoch and epoch < end_epoch:
                if self.option == 6:
                   lambd = -1 * round(max(min(self.freq/(end_epoch-start_epoch) * math.ceil((epoch-start_epoch-1) / self.freq) + self.init_lambd, 1.), self.init_lambd), 4)               
                elif self.option == 0 or self.option == 5:
                    lambd = round(max(min(self.freq/(end_epoch-start_epoch) * math.ceil((epoch-start_epoch-1) / self.freq) + self.init_lambd, 1.), self.init_lambd), 4) 
                else:
                    lambd = round(min(1 - alpha + 0.1, 1.), 2)
            elif epoch >= end_epoch:
                if self.option == 6:
                    lambd = -1.
                else:
                    lambd = 1.0
        else:
            alpha, lambd = self.alpha, self.lambd
        # print(f'during rev in training: {self.training}, alpha: {alpha}, lambda: {lambd}, start_epoch: {start_epoch}, end_epoch: {end_epoch}') # TODO: DEBUG
        return alpha, lambd
    
    def lambd_alpha_options(self, option=0, current=-1, start=0, end=10, freq=10., freeze_ratio=0.6):
        '''
                -1: only change lambda 
                0: linearly change lamba and alpha, interleave 
                5: change alpha linearly first, and then lambd linearly 
        '''
        lambd_min = 0.1
        lambd_max = 0.1 + freeze_ratio
        alpha_min = 1. - freeze_ratio
        alpha_max = 1.
        # breakpoint()
        if option == 0:
            lambd = round(max(min((freq/(end-start) * math.ceil((current-start-1) / freq) + lambd_min), lambd_max), lambd_min), 4) 
            alpha = round(min(max(-round(freq/(end-start), 12) * math.ceil((current-start-1-freq//2) / freq) + 1, alpha_min), alpha_max), 4)
        # elif option == -1:
        #     lambd = -1 * round(max(min(freq/(end-start) * math.ceil((current-start-1) / freq) + lambd_min, lambd_max), lambd_min), 4) 
        #     alpha = 1.0
        elif option == -1:
            alpha = self.alpha 
            lambd = self.lambd
        elif option == 5:
            freq = int(freq/2)
            end_half = start + int((end - start) / 2)
            if current < end_half:
                lambd = lambd_min
            else:
                lambd = round(max(min(freq/(end-end_half) * math.ceil((current-end_half-1) / freq) + lambd_min, lambd_max), lambd_min), 4) 
            alpha = round(min(max(-round(freq/(end_half-start), 12) * math.ceil((current-start-1-freq//2) / freq) + 1, alpha_min), alpha_max), 4)
        elif option == 6:
            freq = int(freq/2)
            end_half = start + int((end - start) / 2) 

            lambd = round(max(min((freq/(end_half-start) * math.ceil((current-start-1) / freq) + lambd_min), lambd_max), lambd_min), 4)    
            if current < end_half:
                alpha = alpha_max   
            else:
                alpha = round(min(max(-round(freq/(end-end_half), 12) * math.ceil((current-end_half-1-freq//2) / freq) + 1, alpha_min), alpha_max), 4)
        return lambd, alpha

    def get_alpha_lambd(self, epoch=-1, iter=-1, 
                        option=0, 
                        start_epoch=1, end_epoch=50, 
                        freq=10., freeze_ratio=0.6, 
                        use_iter=True, 
                        num_training_steps_per_epoch=1000, 
                        stop_epoch=40 # FIXME: not supported yet.
                        ):
        '''
            freq: frequency in epoch if use_iter else frequence in iteration
        '''
        if option == -1:
            alpha = self.alpha 
            lambd = self.lambd
        else:
            if use_iter:
                current = iter
                start = (start_epoch - 1) * num_training_steps_per_epoch
                end = (end_epoch - 1) * num_training_steps_per_epoch
            else:
                current = epoch
                start = start_epoch 
                end = end_epoch 

            if current < start:  # Use original non-reversible for the initial epoch
                alpha = 1.0
                lambd = 0.
            elif current >= start :
                lambd, alpha = self.lambd_alpha_options(option, current, start, end, freq, freeze_ratio)
        return alpha, lambd

    def forward(self, p, x=None, epoch=-1, step=-1):
        if hasattr(p, 'keys'): 
            p, x, epoch, step = p['pos'], p['x'] if 'x' in p.keys() else None, p['epoch'] if 'epoch' in p.keys() else -1, p['iter'] if 'iter' in p.keys() else -1
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
        p_list, x_list = self.patch_embed(p, x)
        center_p, x = p_list[-1], self.proj(x_list[-1].transpose(1, 2))
        pos_embed = self.pos_embed(center_p)

        pos_embed = [self.cls_pos.expand(x.shape[0], -1, -1), pos_embed]
        tokens = [self.cls_token.expand(x.shape[0], -1, -1), x]
        if self.dist_token is not None:
            pos_embed.insert(1, self.dist_pos.expand(x.shape[0], -1, -1)) 
            tokens.insert(1, self.dist_token.expand(x.shape[0], -1, -1)) 
        pos_embed = torch.cat(pos_embed, dim=1)
        x = torch.cat(tokens, dim=1)
        
        with torch.set_grad_enabled(epoch>=self.start_epoch):
            alpha, lambd = self.get_alpha_lambd(epoch, step, 
                        option=self.option, 
                        start_epoch=self.start_epoch, end_epoch=self.end_epoch, 
                        freq=self.freq, freeze_ratio=self.freeze_ratio, 
                        use_iter=self.use_step, 
                        num_training_steps_per_epoch=self.num_training_steps_per_epoch, 
                        stop_epoch=self.stop_epoch
                                                ) 
            if self.add_pos_each_block:
                if use_inv:
                    x = self._use_inv(x, pos_embed, alpha, lambd, num_cached=self.num_cached)
                else:
                    for block in self.blocks:
                        x = block(x, pos_embed)
            else:
                x = self.pos_drop(x + pos_embed)
                if use_inv:
                    x = self._use_inv(x, None, alpha, lambd, num_cached=self.num_cached)
                else:
                    for block in self.blocks:
                        x = block(x)
        x = self.norm(x)
        return p_list, x_list, x

    def _use_inv(self, x, pos_embed=None, alpha=0., lambd=1., num_cached=0):
        x = torch.cat((x, x), dim=-1)
        # breakpoint()
        if self.use_customized_backprop:    
            x = RevBackProp.apply(x, self.blocks, pos_embed, alpha, lambd, num_cached)
        else:
            for block in self.blocks:
                x = block(x, pos_embed)            
        C = int(x.shape[-1] / 2)
        x = (x[:,:,:C] + x[:,:,C:]) / 2     
        return x   

    def forward_cls_feat(self, p, x=None):  # p: p, x: features
        _, _, x = self.forward(p, x)
        token_features = x[:, self.n_tokens:, :]
        cls_feats = []
        for token_type in self.global_feat:
            if 'cls' in token_type:
                cls_feats.append(x[:, 0, :])
            elif 'max' in token_type:
                cls_feats.append(torch.max(token_features, dim=1, keepdim=False)[0])
            elif token_type in ['avg', 'mean']:
                cls_feats.append(torch.mean(token_features, dim=1, keepdim=False))
        global_features = torch.cat(cls_feats, dim=1)
        
        if self.dist_token is not None and self.training:
            return global_features, x[:, 1, :]
        else: 
            return global_features

    def forward_seg_feat(self, p, x=None, epoch=-1):  # p: p, x: features
        p_list, x_list, x = self.forward(p, x, epoch)
        x_list[-1] = x.transpose(1, 2)
        return p_list, x_list


@MODELS.register_module()
class InvPointViTDecoder(nn.Module):
    """ Decoder of Point Vision Transformer for segmentation.
    """
    def __init__(self,
                 encoder_channel_list: List[int], 
                 decoder_layers: int = 2, 
                 n_decoder_stages: int = 2, # TODO: ablate this 
                 scale: int = 4,
                 channel_scaling: int = 1,  
                 sampler: str = 'fps',
                 global_feat=None, 
                 progressive_input=False,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        if global_feat is not None:
            self.global_feat = global_feat.split(',')
            num_global_feat = len(self.global_feat)
        else:
            self.global_feat = None
            num_global_feat = 0 
 
        self.in_channels = encoder_channel_list[-1]
        self.scale = scale
        self.n_decoder_stages = n_decoder_stages
        
        if progressive_input:
            skip_dim = [self.in_channels//2**i for i in range(n_decoder_stages-1, 0, -1)]
        else:
            skip_dim = [0 for i in range(n_decoder_stages-1)]
        skip_channels = [encoder_channel_list[0]] + skip_dim
        
        fp_channels = [self.in_channels*channel_scaling]
        for _ in range(n_decoder_stages-1):
           fp_channels.insert(0, fp_channels[0] * channel_scaling) 
        
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages-1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages] * (num_global_feat + 1)

        if sampler.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif sampler.lower() == 'random':
            self.sample_fn = random_sample
            
    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
                [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        """
        Args:
            p (List(Tensor)): List of tensor for p, length 2, input p and center p
            f (List(Tensor)): List of tensor for feature maps, input features and out features
        """
        if len(p) != (self.n_decoder_stages + 1):
            for i in range(self.n_decoder_stages - 1): 
                pos = p[i] 
                idx = self.sample_fn(pos, pos.shape[1] // self.scale).long()
                new_p = torch.gather(pos, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                p.insert(1, new_p)
                f.insert(1, None)
        cls_token = f[-1][:, :, 0:1]
        f[-1] = f[-1][:, :, 1:].contiguous()
        
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        f_out = f[-len(self.decoder) - 1] 
        
        if self.global_feat is not None:
            global_feats = []
            for token_type in self.global_feat:
                if 'cls' in token_type:
                    global_feats.append(cls_token)
                elif 'max' in token_type:
                    global_feats.append(torch.max(f_out, dim=2, keepdim=True)[0])
                elif token_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(f_out, dim=2, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, f_out.shape[-1])
            f_out = torch.cat((global_feats, f_out), dim=1)
        return f_out 


@MODELS.register_module()
class InvPointViTPartDecoder(nn.Module):
    """ Decoder of Point Vision Transformer for segmentation.
    """
    def __init__(self,
                 encoder_channel_list: List[int], 
                 decoder_layers: int = 2, 
                 n_decoder_stages: int = 2,
                 scale: int = 4,
                 channel_scaling: int = 1,  
                 sampler: str = 'fps',
                 global_feat=None, 
                 progressive_input=False,
                 cls_map='pointnet2',
                 num_classes: int = 16,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        if global_feat is not None:
            self.global_feat = global_feat.split(',')
            num_global_feat = len(self.global_feat)
        else:
            self.global_feat = None
            num_global_feat = 0 
 
        self.in_channels = encoder_channel_list[-1]
        self.scale = scale
        self.n_decoder_stages = n_decoder_stages
        
        if progressive_input:
            skip_dim = [self.in_channels//2**i for i in range(n_decoder_stages-1, 0, -1)]
        else:
            skip_dim = [0 for i in range(n_decoder_stages-1)]
        skip_channels = [encoder_channel_list[0]] + skip_dim
        
        fp_channels = [self.in_channels*channel_scaling]
        for _ in range(n_decoder_stages-1):
           fp_channels.insert(0, fp_channels[0] * channel_scaling) 

        self.cls_map = cls_map
        self.num_classes = num_classes
        act_args = kwargs.get('act_args', {'act': 'relu'}) 
        if self.cls_map == 'curvenet':
            # global features
            self.global_conv2 = nn.Sequential(
                create_convblock1d(fp_channels[-1] * 2, 128,
                                   norm_args=None,
                                   act_args=act_args))
            self.global_conv1 = nn.Sequential(
                create_convblock1d(fp_channels[-2] * 2, 64,
                                   norm_args=None,
                                   act_args=act_args))
            skip_channels[0] += 64 + 128 + 16  # shape categories labels
        elif self.cls_map == 'pointnet2':
            self.convc = nn.Sequential(create_convblock1d(16, 64,
                                                          norm_args=None,
                                                          act_args=act_args))
            skip_channels[0] += 64  # shape categories labels
            
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages-1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages] * (num_global_feat + 1)

        if sampler.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif sampler.lower() == 'random':
            self.sample_fn = random_sample
            
    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
                [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f, cls_label):
        """
        Args:
            p (List(Tensor)): List of tensor for p, length 2, input p and center p
            f (List(Tensor)): List of tensor for feature maps, input features and out features
        """
        if len(p) != (self.n_decoder_stages + 1):
            for i in range(self.n_decoder_stages - 1): 
                pos = p[i] 
                idx = self.sample_fn(pos, pos.shape[1] // self.scale).long()
                new_p = torch.gather(pos, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                p.insert(1, new_p)
                f.insert(1, None)
        cls_token = f[-1][:, :, 0:1]
        f[-1] = f[-1][:, :, 1:].contiguous()
        
        B, N = p[0].shape[0:2]
        if self.cls_map == 'pointnet2':
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
            cls_one_hot = self.convc(cls_one_hot)
             
        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        i = -len(self.decoder) 
        f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], torch.cat([cls_one_hot, f[i - 1]], 1)], [p[i], f[i]])])[1]

        f_out = f[-len(self.decoder) - 1] 
        
        if self.global_feat is not None:
            global_feats = []
            for token_type in self.global_feat:
                if 'cls' in token_type:
                    global_feats.append(cls_token)
                elif 'max' in token_type:
                    global_feats.append(torch.max(f_out, dim=2, keepdim=True)[0])
                elif token_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(f_out, dim=2, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, f_out.shape[-1])
            f_out = torch.cat((global_feats, f_out), dim=1)
        return f_out 
