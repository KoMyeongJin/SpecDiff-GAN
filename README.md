<table>
    <tr>
        <td>
            <p align="center">
                <img src="img/SpecDiff-GAN Architecture.png" width="80%"><br>
                <b>SpecDiff-GAN Architecture</b>
            </p>
        </td>
    </tr>
</table>

# SpecDiff-GAN

[![githubio](https://img.shields.io/static/v1?message=Audio%20Samples&logo=Github&labelColor=grey&color=blue&logoColor=white&label=%20&style=flat-square)](https://komyeongjin.github.io/SpecDiff-GAN-Demo/) [![GitHub](https://img.shields.io/github/license/KoMyeongjin/SpecDiff-GAN?style=flat-square)](./licence)

#### PyTorch implementation of [Adversarial Training of Denoising Diffusion Model Using Dual Discriminators for High-Fidelity Multi-Speaker TTS]()

This project is based on [keonlee9420's implementation](https://github.com/keonlee9420/DiffGAN-TTS/) of DiffGAN-TTS


### Audio Samples
Audio samples are available at the following [link](https://komyeongjin.github.io/SpecDiff-GAN-Demo/).

## Getting Started

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

### Inference
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --speaker_id SPEAKER_ID --restore_step RESTORE_STEP --mode single --dataset DATASET
```

The dictionary of learned speakers can be found at `preprocessed_data/DATASET/speakers.json`, and the generated utterances will be put in `output/result/`.

### Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/DATASET/val.txt --restore_step RESTORE_STEP --mode batch --dataset DATASET
```
to synthesize all utterances in ``preprocessed_data/DATASET/val.txt``.


### Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --model MODEL --restore_step RESTORE_STEP --mode single --dataset DATASET --duration_control 0.8 --energy_control 0.8
```

Please note that the controllability is originated from [FastSpeech2](https://arxiv.org/abs/2006.04558) and not a vital interest of this model.

## Training

### Datasets

The supported dataset is

- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443): The CSTR VCTK Corpus includes speech data uttered by 110 English speakers (**multi-speaker TTS**) with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive.

You can add another dataset with modifying config files in /config/VCTK
If you want to train Single speaker dataset, please change `multi_speaker` option in model.yaml

### Preprocessing

- Before you run preprocess code, please set your dataset's location to `corpus_path` option in preprocess.yaml
- For a **multi-speaker TTS** with external speaker embedder, download [ResCNN Softmax+Triplet pretrained model](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP) of [philipperemy's DeepSpeaker](https://github.com/philipperemy/deep-speaker) for the speaker embedding and locate it in `./deepspeaker/pretrained_models/`.
- Run
    ```
    python3 prepare_align.py --dataset DATASET
    ```
    for some preparations.

    For the forced alignment, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
    Pre-extracted alignments for the datasets are provided [here](https://drive.google.com/file/d/1racLb2K84B-dJgdzXpi5EBpXFrOKQdg6/). 
    You have to unzip the files in `preprocessed_data/DATASET/TextGrid/`. Alternately, you can [run the aligner by yourself](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/index.html).

    After that, run the preprocessing script by
    ```
    python3 preprocess.py --dataset DATASET
    ```

### Training

Train with

```
python3 train.py --dataset DATASET
```

- Restore training status:

    To restore training status from checkpoint, you must pass `--restore_step` with the step of auxiliary FastSpeech2 training as the following command.
    ```
    python3 train.py --restore_step RESTORE_STEP --dataset DATASET
    ```
    For example, if the last checkpoint is saved at 200000 steps during the training, you have to set `--restore_step` with `200000`. Then it will load and freeze the aux model and then continue the training under the active shallow diffusion mechanism.

### TensorBoard

Use
```
tensorboard --logdir output/log/DATASET
```
to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

## Notes

- In addition to the Diffusion Decoder, the Variance Adaptor is also conditioned on speaker information.
- Two options for embedding for the **multi-speaker TTS** setting: training speaker embedder from scratch or using a pre-trained [philipperemy's DeepSpeaker](https://github.com/philipperemy/deep-speaker) model (as [STYLER](https://github.com/keonlee9420/STYLER) did). You can toggle it by setting the config (between `'none'` and `'DeepSpeaker'`).

:warning: DiffusionDiscrimiantor's code is Licenced with [Nvidia Source Code License](https://github.com/NVlabs/denoising-diffusion-gan), please check before you use this code, it is important to adhere to the terms of their license.

## Citation

```bibtex
@article{
    title={Adversarial Training of Denoising Diffusion Model Using Dual Discriminators for High-Fidelity Multi-Speaker TTS},
    author={Ko, Myeongjin and Choi, Yong-Hoon},
    journal={},
    year={2023}
}
```

## References
### Codes
- [keonlee9420's DiffGAN-TTS](https://github.com/keonlee9420/DiffSinger)
- [mindslab-ai's univnet](https://github.com/mindslab-ai/univnet)
- [NVlabs' denoising-diffusion-gan](https://github.com/NVlabs/denoising-diffusion-gan) 
### Papers
- [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889)
- [Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://arxiv.org/abs/2112.07804)
- [DiffGAN-TTS: High-Fidelity and Efficient Text-to-Speech with Denoising Diffusion GANs](https://arxiv.org/abs/2201.11972)
### Datasets
- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)
