# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 00:27:55 2026

@author: Zamikhaya.Magogotya
"""

"""
==========================================================

SAT-RADAR AI

Vision Transformer
Satellite → Radar

Author : Zamikhaya Magogotya

==========================================================
"""

import torch
import torch.nn as nn


###########################################################
# Patch Embedding
###########################################################

class PatchEmbedding(nn.Module):

    def __init__(
        self,
        in_channels=8,
        embed_dim=256,
        patch_size=16,
    ):

        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):

        x = self.projection(x)

        B, C, H, W = x.shape

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x, H, W


###########################################################
# Transformer Encoder
###########################################################

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        heads=8,
        depth=6,
        mlp_ratio=4,
        dropout=0.1,
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=heads,

            dim_feedforward=embed_dim * mlp_ratio,

            dropout=dropout,

            batch_first=True,

        )

        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=depth,

        )

    def forward(self, x):

        return self.encoder(x)


###########################################################
# Decoder
###########################################################

class RadarDecoder(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        out_channels=3,
        image_size=256,
        patch_size=16,
    ):

        super().__init__()

        self.embed_dim = embed_dim

        self.image_size = image_size

        self.patch_size = patch_size

        self.grid = image_size // patch_size

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                embed_dim,
                128,
                4,
                2,
                1,
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1,
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1,
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                out_channels,
                4,
                2,
                1,
            ),
        )

    def forward(self, x):

        B = x.shape[0]

        x = x.transpose(1, 2)

        x = x.reshape(

            B,

            self.embed_dim,

            self.grid,

            self.grid,

        )

        x = self.decoder(x)

        return x


###########################################################
# Full Model
###########################################################

class SatRadarViT(nn.Module):

    def __init__(

        self,

        in_channels=8,

        out_channels=3,

        embed_dim=256,

        patch_size=16,

        image_size=256,

        depth=6,

        heads=8,

    ):

        super().__init__()

        self.embedding = PatchEmbedding(

            in_channels,

            embed_dim,

            patch_size,

        )

        self.encoder = TransformerEncoder(

            embed_dim,

            heads,

            depth,

        )

        self.decoder = RadarDecoder(

            embed_dim,

            out_channels,

            image_size,

            patch_size,

        )

    def forward(self, x):

        x, H, W = self.embedding(x)

        x = self.encoder(x)

        x = self.decoder(x)

        return x