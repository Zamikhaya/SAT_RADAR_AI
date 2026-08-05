# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 01:19:59 2026

@author: Zamikhaya.Magogotya
"""

"""
==============================================================

SAT-RADAR AI

Temporal Vision Transformer

Author : Zamikhaya Magogotya

==============================================================
"""

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Input

    (Batch, Time, Channels, Height, Width)

    Example

    (4,5,8,256,256)

    """

    def __init__(self,
                 image_size=256,
                 patch_size=16,
                 in_channels=8,
                 embed_dim=512,
                 depth=8,
                 num_heads=8,
                 num_frames=5):

        super().__init__()

        self.num_frames = num_frames

        ####################################################
        # Spatial Patch Embedding
        ####################################################

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.num_patches = (image_size // patch_size) ** 2

        ####################################################
        # Spatial Position Embedding
        ####################################################

        self.spatial_position = nn.Parameter(
            torch.randn(
                1,
                self.num_patches,
                embed_dim
            )
        )

        ####################################################
        # Temporal Embedding
        ####################################################

        self.temporal_position = nn.Parameter(
            torch.randn(
                1,
                num_frames,
                embed_dim
            )
        )

        ####################################################
        # Transformer
        ####################################################

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        ####################################################
        # Decoder
        ####################################################

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                embed_dim,
                256,
                2,
                2
            ),

            nn.GELU(),

            nn.ConvTranspose2d(
                256,
                128,
                2,
                2
            ),

            nn.GELU(),

            nn.ConvTranspose2d(
                128,
                64,
                2,
                2
            ),

            nn.GELU(),

            nn.ConvTranspose2d(
                64,
                32,
                2,
                2
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                3,
                1
            )

        )

    ########################################################

    def forward(self, x):

        """
        x

        (B,T,C,H,W)

        """

        B, T, C, H, W = x.shape

        tokens = []

        ####################################################
        # Encode each satellite image
        ####################################################

        for t in range(T):

            frame = x[:, t]

            patches = self.patch_embed(frame)

            patches = patches.flatten(2)

            patches = patches.transpose(1, 2)

            patches = patches + self.spatial_position

            patches = patches + self.temporal_position[:, t].unsqueeze(1)

            tokens.append(patches)

        ####################################################
        # Stack all temporal tokens
        ####################################################

        tokens = torch.cat(tokens, dim=1)

        ####################################################
        # Transformer
        ####################################################

        tokens = self.transformer(tokens)

        ####################################################
        # Average over time
        ####################################################

        tokens = tokens.reshape(
            B,
            T,
            self.num_patches,
            -1
        )

        tokens = tokens.mean(1)

        ####################################################
        # Convert tokens back to image
        ####################################################

        size = int(self.num_patches ** 0.5)

        tokens = tokens.transpose(1, 2)

        tokens = tokens.reshape(
            B,
            -1,
            size,
            size
        )

        ####################################################
        # Decoder
        ####################################################

        output = self.decoder(tokens)

        return output