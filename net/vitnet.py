# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import  torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from net.pos_embed import PositionalEncoding
from config import config as cfg


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cplinear1 = nn.Linear(3, embed_dim)
        self.plinear1 = nn.Linear(3, embed_dim)


        self.encoder_embed = nn.Parameter(torch.zeros(1, cfg.teeth_nums, embed_dim), requires_grad=False)
        self.decoder_embed = nn.Parameter(torch.zeros(1, cfg.teeth_nums, embed_dim*5), requires_grad=False)


        self.teeth1 = nn.Parameter(torch.zeros(cfg.teeth_nums, 1, 1),requires_grad=True)
        self.teeth_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])


        self.cp_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim*5)

        self.tb_props = nn.ModuleList([
            Block(embed_dim*5, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])


        self.linear21 = nn.Linear(embed_dim*5, embed_dim)
        self.linear22 = nn.Linear(embed_dim, 3)
        self.linear23 = nn.Linear(embed_dim, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed1 = PositionalEncoding(self.encoder_embed.shape[1], self.encoder_embed.shape[2], self.device)
        self.encoder_embed.data.copy_(pos_embed1.float().unsqueeze(0))

        pos_embed2 = PositionalEncoding(self.decoder_embed.shape[1], self.decoder_embed.shape[2], self.device)
        self.decoder_embed.data.copy_(pos_embed2.float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        TB, C, N = x.shape
        x = x.permute(0, 2, 1)
        xc = self.cplinear1(x[:, 0:1,3 :]).permute(1, 0, 2)
        x = self.plinear1(x[:, :, : 3])

        #tcp
        xc = self.encoder_embed + xc
        for cpb in self.cp_blocks:
            xc = cpb(xc)

        # apply Transformer blocks
        teeths = []
        for blk in self.teeth_blocks:
            x = blk(x)
            teeths.append(x)

        x = torch.cat(teeths, dim=-1)
        x = torch.mean(x, dim=1, keepdim=True)
        x = x.permute(1, 0, 2)
        x = torch.cat([x, xc],  dim=-1)
        x = x #+ self.decoder_embed
        x_ = x.clone().permute(1, 0, 2)
        for blk in self.tb_props:
            x = blk(x)

        x = x.permute(1, 0, 2)
        x = self.teeth1 * x + x_

        x = torch.mean(x, dim=1)
        x = F.relu(self.linear21(x))
        transv = 10*F.tanh(self.linear22(x))
        x = F.tanh(self.linear23(x))
        dofx = torch.nn.functional.normalize(x, dim=-1)

        return dofx, transv

    def forward(self, imgs, mask_ratio=0.75):
        dofx, transv = self.forward_encoder(imgs)

        return dofx, transv

def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(embed_dim=256, depth=4, num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model





