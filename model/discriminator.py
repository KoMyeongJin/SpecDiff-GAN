import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from model.disc_modules.modules import TimestepEmbedding, DownConvBlock
from model.disc_modules.dense_layer import dense, conv2d


class DiffusionDiscriminator(nn.Module):
    """ Originated from https://github.com/rosinality/stylegan2-pytorch
    To view a copy of this license (MIT), visit
    https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE
    """
    def __init__(self, nc=1, ngf=32, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv2 = DownConvBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv3 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv4 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv5 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv6 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
        self.end_linear = dense(ngf * 8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, x_t, t):
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        input_x = input_x.unsqueeze(1)

        h = self.start_conv(input_x)
        h = self.conv1(h, t_embed)

        h = self.conv2(h, t_embed)

        h = self.conv3(h, t_embed)
        h = self.conv4(h, t_embed)
        h = self.conv5(h, t_embed)

        out = self.conv6(h, t_embed)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)

        return out


class SpectrogramDiscriminator(torch.nn.Module):
    def __init__(self, model_config, resolution=None):
        super(SpectrogramDiscriminator, self).__init__()

        self.LRELU_SLOPE = model_config["spec_discriminator"]["lReLU_slope"]
        self.multi_speaker = model_config["multi_speaker"]
        residual_channels = model_config["denoiser"]["residual_channels"]

        norm_f = (
            weight_norm
            if model_config["spec_discriminator"]["use_spectral_norm"] is False
            else spectral_norm
        )
        self.conv_prev = norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            ]
        )
        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(norm_f(nn.Linear(residual_channels, 32)))
        self.conv_post = nn.ModuleList(
            [
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
                norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1))),
            ]
        )

    def forward(self, x, s):
        fmap = []

        x = x.unsqueeze(1)

        x = self.conv_prev(x)
        x = F.leaky_relu(x, self.LRELU_SLOPE)
        fmap.append(x)

        if self.multi_speaker:
            speaker_emb = (
                self.spk_mlp(s).unsqueeze(-1).expand(-1, -1, x.shape[-2]).unsqueeze(-1)
            )
            x = x + speaker_emb

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post[0](x)
        x = F.leaky_relu(x, self.LRELU_SLOPE)
        x = self.conv_post[1](x)

        x = torch.flatten(x, 1, -1)

        return fmap, x
