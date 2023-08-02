# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, visit
# https://github.com/NVlabs/denoising-diffusion-gan/blob/main/LICENSE
# ---------------------------------------------------------------
import torch.nn as nn
import numpy as np

from model.disc_modules.dense_layer import dense, conv2d
from model.disc_modules.layers import get_timestep_embedding as get_sinusoidal_positional_embedding
from model.disc_modules import up_or_down_sampling


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


# %%
class DownConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            t_emb_dim=128,
            downsample=False,
            act=nn.LeakyReLU(0.2),
            fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Sequential(
            conv2d(in_channel, out_channel, kernel_size, padding=padding),
        )

        self.conv2 = nn.Sequential(
            conv2d(out_channel, out_channel, kernel_size, padding=padding, init_scale=0.)
        )
        self.dense_t1 = dense(t_emb_dim, out_channel)

        self.act = act

        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False),
        )

    def forward(self, input, t_emb):
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]

        out = self.act(out)

        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out
