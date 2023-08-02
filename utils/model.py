import os
import json

import torch
import numpy as np

import hifigan
import itertools
from model import SpecDiffGAN, ScheduledOptim, SpectrogramDiscriminator, DiffusionDiscriminator


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    epoch = 1
    model = SpecDiffGAN(args, preprocess_config, model_config, train_config).to(device)
    diff_d = DiffusionDiscriminator().to(device)
    spec_d = SpectrogramDiscriminator(model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        epoch = int(ckpt["epoch"])
        model.load_state_dict(ckpt["G"])  # named_parameters: {'variance_adaptor', 'diffusion', 'mel_linear', 'text_encoder', 'decoder'}
        diff_d.load_state_dict(ckpt["D_Diff"])  # named_parameters: {'input_projection', 'cond_conv_block', 'uncond_conv_block', 'conv_block', 'mlp'}
        spec_d.load_state_dict(ckpt["D_Spec"])

    if train:
        init_lr_G = train_config["optimizer"]["init_lr_G"]
        init_lr_D = train_config["optimizer"]["init_lr_D"]
        betas = train_config["optimizer"]["betas"]
        gamma = train_config["optimizer"]["gamma"]
        optG_fs2 = ScheduledOptim(model, train_config, model_config, args.restore_step)
        optG = torch.optim.Adam(model.parameters(), lr=init_lr_G, betas=betas)
        optD = torch.optim.AdamW(itertools.chain(diff_d.parameters(), spec_d.parameters()), lr=init_lr_D, betas=betas)
        sdlG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma)
        sdlD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma)
        if args.restore_step:
            optG_fs2.load_state_dict(ckpt["optG_fs2"])
            optG.load_state_dict(ckpt["optG"])
            optD.load_state_dict(ckpt["optD"])
            sdlG.load_state_dict(ckpt["sdlG"])
            sdlD.load_state_dict(ckpt["sdlD"])
        model.train()
        diff_d.train()
        spec_d.train()
        return model, diff_d, spec_d, optG_fs2, optG, optD, sdlG, sdlD, epoch

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_netG_params(model_kernel):
    return list(model_kernel.C.parameters()) \
        + list(model_kernel.Z.parameters()) \
        + list(model_kernel.G.parameters())


def get_netD_params(model_kernel):
    return model_kernel.D.parameters()


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config_universal.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
