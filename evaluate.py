import torch
from torch.utils.data import DataLoader

from utils.tools import to_device, log, synth_one_sample
from model import SpecDiffGANLoss
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args, model, diff_d, spec_d, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = SpecDiffGANLoss(args, preprocess_config, model_config, train_config).to(device)

    loss_sums = [{k : 0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)

            with torch.no_grad():
                #######################
                # Evaluate Discriminator #
                #######################

                # Forward
                output, *_ = model(*(batch[2:]))

                xs, spk_emb, t, mel_masks = *(output[1:4]), output[9]
                x_ts, x_t_prevs, x_t_prev_preds, spk_emb, t = \
                    [x.detach() if x is not None else x for x in (list(xs) + [spk_emb, t])]

                D_fake_diff = diff_d(x_ts, x_t_prev_preds, t)
                D_real_diff = diff_d(x_ts, x_t_prevs, t)
                D_fake_spec_feats, D_fake_spec = spec_d(output[0], spk_emb)
                D_real_spec_feats, D_real_spec = spec_d(batch[6], spk_emb)

                D_loss_real, D_loss_fake = Loss.d_loss_fn(D_real_diff[-1], D_real_spec, D_fake_diff[-1], D_fake_spec)

                D_loss = D_loss_real + D_loss_fake

                #######################
                # Evaluate Generator #
                #######################

                # Forward
                output, p_targets, coarse_mels = model(*(batch[2:]))
                # Update Batch
                batch[9] = p_targets

                (x_ts, x_t_prevs, x_t_prev_preds), spk_emb, t, mel_masks = *(output[1:4]), output[9]

                D_fake_diff = diff_d(x_ts, x_t_prev_preds, t)
                D_real_diff = diff_d(x_ts, x_t_prevs, t)
                D_fake_spec_feats, D_fake_spec = spec_d(output[0], spk_emb)
                D_real_spec_feats, D_real_spec = spec_d(batch[6], spk_emb)

                adv_loss = Loss.g_loss_fn(D_fake_diff[-1], D_fake_spec)

                (
                    fm_loss,
                    recon_loss,
                    mel_loss,
                    pitch_loss,
                    energy_loss,
                    duration_loss,
                ) = Loss(
                    model,
                    batch,
                    output,
                    coarse_mels,
                    (D_real_diff, D_real_spec_feats, D_fake_diff, D_fake_spec_feats),
                )

                G_loss = recon_loss + fm_loss + adv_loss

                losses = [D_loss + G_loss, D_loss, G_loss, recon_loss, fm_loss, adv_loss, mel_loss, pitch_loss, energy_loss, duration_loss]

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_msg = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k : v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_msg.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_msg.append(loss_sum / len(dataset))
    loss_means_msg = loss_means_msg[0:2] + loss_means_msg[5:]

    message = "Validation Step {}, Total Loss: {:.4f}, D_loss: {:.4f}, adv_loss: {:.4f}, mel_loss: {:.4f}, pitch_loss: {:.4f}, energy_loss: {:.4f}, duration_loss: {:.4f}".format(
        *([step] + [loss_mean for loss_mean in loss_means_msg])
    )

    figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
        args,
        batch,
        output,
        coarse_mels,
        vocoder,
        model_config,
        preprocess_config,
        model.diffusion,
    )
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    if logger is not None:
        log(logger, step, losses=loss_means)
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        log(
            logger,
            step,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/reconstructed",
        )
        log(
            logger,
            step,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/synthesized",
        )

    return message
