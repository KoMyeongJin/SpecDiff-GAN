import argparse
import os

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample
from model import SpecDiffGANLoss
from dataset import Dataset

from evaluate import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", args, preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, diff_d, spec_d, optG_fs2, optG, optD, sdlG, sdlD, epoch = get_model(args, configs, device, train=True)
    num_params_G = get_param_num(model)
    num_params_D_Diff = get_param_num(diff_d)
    num_params_D_Spec = get_param_num(spec_d)
    Loss = SpecDiffGANLoss(args, preprocess_config, model_config, train_config).to(device)
    print("Number of SpecDiff-GAN Parameters            :", num_params_G)
    print("          DiffusionDiscriminator Parameters  :", num_params_D_Diff)
    print("          SpectrogramDiscriminator Parameters:", num_params_D_Spec)

    print("          All Parameters                     :", num_params_G + num_params_D_Diff + num_params_D_Spec)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    def model_update(model, step, loss, optimizer):
        # Backward
        loss = (loss / grad_acc_step).backward()
        if step % grad_acc_step == 0:
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                #######################
                # Train Discriminator #
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

                D_loss_real, D_loss_fake = Loss.d_loss_fn(D_real_diff[-1], D_real_spec, D_fake_diff[-1],
                                                            D_fake_spec)

                D_loss = D_loss_real + D_loss_fake

                model_update(diff_d, step, D_loss, optD)

                #######################
                # Train Generator #
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

                G_loss = adv_loss + recon_loss + fm_loss
                model_update(model, step, G_loss, optG)

                losses = [D_loss + G_loss, D_loss, G_loss, recon_loss, fm_loss, adv_loss, mel_loss, pitch_loss,
                          energy_loss, duration_loss]
                losses_msg = [D_loss + G_loss, D_loss, adv_loss, mel_loss, pitch_loss, energy_loss, duration_loss]

                if step % log_step == 0:
                    losses_msg = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses_msg]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, D_loss: {:.4f}, adv_loss: {:.4f}, mel_loss: {:.4f}, pitch_loss: {:.4f}, energy_loss: {:.4f}, duration_loss: {:.4f}".format(
                        *losses_msg
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses, lr=sdlG.get_last_lr()[-1])

                if step % synth_step == 0:
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
                    log(
                        train_logger,
                        step,
                        figs=figs,
                        tag="Training",
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        step,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/reconstructed",
                    )
                    log(
                        train_logger,
                        step,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/synthesized",
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(args, model, diff_d, spec_d, step, configs, val_logger, vocoder, losses)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "G": model.state_dict(),
                            "D_Diff": diff_d.state_dict(),
                            "D_Spec": spec_d.state_dict(),
                            "optG_fs2": optG_fs2._optimizer.state_dict(),
                            "optG": optG.state_dict(),
                            "optD": optD.state_dict(),
                            "sdlG": sdlG.state_dict(),
                            "sdlD": sdlD.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step >= total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1
        sdlG.step()
        sdlD.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"] + "_{}".format(path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"] + "_{}".format(path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"] + "_{}".format(path_tag)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt

        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    if model_config["multi_speaker"]:
        print(" ---> Type of Speaker Embedder:", preprocess_config["preprocessing"]["speaker_embedder"])
    print(" ---> Total Batch Size:", int(train_config["optimizer"]["batch_size"]))
    print(" ---> Use Pitch Embed:", model_config["variance_embedding"]["use_pitch_embed"])
    print(" ---> Use Energy Embed:", model_config["variance_embedding"]["use_energy_embed"])
    print(" ---> Path of ckpt:", train_config["path"]["ckpt_path"])
    print(" ---> Path of log:", train_config["path"]["log_path"])
    print(" ---> Path of result:", train_config["path"]["result_path"])
    print("================================================================================================")

    main(args, configs)
