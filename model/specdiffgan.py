import os
import json

import torch.nn as nn

from .modules import FastspeechEncoder, FastspeechDecoder, VarianceAdaptor
from .diffusion import GaussianDiffusion
from utils.tools import get_mask_from_lengths


class SpecDiffGAN(nn.Module):
    def __init__(self, args, preprocess_config, model_config, train_config):
        super(SpecDiffGAN, self).__init__()
        self.model_config = model_config

        self.text_encoder = FastspeechEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.diffusion = GaussianDiffusion(args, preprocess_config, model_config, train_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        mel2phs=None,
        spker_embeds=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.text_encoder(texts, src_masks)

        speaker_emb = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_emb = self.speaker_emb(speakers)  # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_emb = self.speaker_emb(spker_embeds)  # [B, H]

        (
            output,
            p_targets,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            max_src_len,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            mel2phs,
            p_control,
            e_control,
            d_control,
            speaker_emb,
        )

        coarse_mels = None
        (
            output, # x_0_pred
            x_ts,
            x_t_prevs,
            x_t_prev_preds,
            diffusion_step,
        ) = self.diffusion(
            mels,
            output,
            speaker_emb,
            mel_masks,
        )

        return [
            output,
            (x_ts, x_t_prevs, x_t_prev_preds),
            speaker_emb,
            diffusion_step,
            p_predictions,  # cannot detach each value in dict but no problem since loss will not use it
            e_predictions,
            log_d_predictions,  # cannot detach each value in dict but no problem since loss will not use it
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        ], p_targets, coarse_mels
