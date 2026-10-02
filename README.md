# SSR-Net

Code and trained weights for

> **SSR-Net: Damage-aware restoration of Han-Wei inscription rubbings via spatial attention and soft-mask fusion**
> Bing Wu, Qingshuang Dong, Xianpeng Jiao, Changle Yang, Jie Wang. *npj Heritage Science* (under review).

SSR-Net (Spatial-attention and Soft-mask Restoration Network) restores damaged character images from
inscription rubbings. A U-Net encoder–decoder with spatial attention predicts a candidate image *G* and a
damage map *M*; the output replaces only the pixels predicted as damaged:

```
I_out = I_in + sigmoid(M) * (G − I_in)
```

so intact strokes are passed through unchanged, and the damage map records which pixels were reconstructed.

## Contents

| File | Purpose |
|---|---|
| `ssrnet/model.py` | Network definition and soft-mask fusion |
| `ssrnet/data.py` | Paired data, character-level train/validation split, reference selection |
| `ssrnet/losses.py` | Loss functions, multi-scale discriminator (stage 1), EMA |
| `configs/stage1.json`, `configs/stage2.json` | Training configurations used in the paper |
| `train.py` | Training (both stages) |
| `test.py` | Restoration of damaged images |
| `evaluate.py` | PSNR, SSIM, LPIPS and damaged-region PSNR |

The rubbing images used in the paper were obtained under licence from a commercial database and cannot be
redistributed; this repository therefore contains no images.

## Installation

Python 3.9 or later and PyTorch 2.0 or later.

```
pip install -r requirements.txt
```

## Trained weights

Download `ssrnet_weights.pth` (86.7 MB) from the [Releases](../../releases) page of this repository.
SHA-256: `acba108fa31cb6a5e867d4e73962b731fc8abf492fe858e508985cec7a0edb4f`.

These are the weights evaluated in the paper (exponential moving average, selected on the validation split).

## Restoring images

```
python test.py --weights ssrnet_weights.pth --input damaged/ --ref references/ --out restored/ --save-mask
```

Images are processed at 256×256. Each damaged character is restored together with a reference character,
an intact character image, preferably from the same stele. `--ref` may be one image used for all inputs or
a folder: a file with the same name as the input is used if it exists, otherwise a reference is drawn at
random from the folder (`--seed`). A CSV with columns `target,reference` (`--pairs`) fixes the pairing. In
the paper, the choice of reference changed the results by at most 0.02 dB.

With `--save-mask`, the predicted damage map is written next to each restoration (`*_damage.png`, white =
reconstructed).

For real rubbings, the damaged regions are marked by a specialist and standardized before restoration, as
described in the paper.

## Training

Training uses pairs of degraded and clean character images with identical file names:

```
pairs/Input/<stele>_<character>_<a>_<b>.png     degraded
pairs/Target/<stele>_<character>_<a>_<b>.png    clean
```

The first underscore-separated field names the stele; the file name without its last two fields identifies
the source character. Ten percent of the source characters (seed 20260913) form the validation split, so
that all degraded versions of a character fall into the same split. References are drawn from training
characters of the same stele, never from the same character.

The model in the paper was trained in two stages on one NVIDIA L4 GPU:

```
# stage 1: from random initialization, with a multi-scale LSGAN discriminator
python train.py --stage 1 --data pairs/ --out runs/stage1

# stage 2: fine-tuning of the best stage-1 weights, without the discriminator
python train.py --stage 2 --data pairs/ --out runs/stage2 --init runs/stage1/best_ema.pth
```

After every epoch the EMA weights are evaluated on the validation split, and the weights with the highest
validation PSNR are saved as `best_ema.pth`. Each run also writes `log.csv`, `split_manifest.csv` and the
configuration used. Hyperparameters and loss weights are in `configs/`.

## Evaluation

```
python evaluate.py --pred restored/ --gt ground_truth/ --input damaged/ --csv per_image.csv
```

PSNR and SSIM are computed with scikit-image on 8-bit RGB images, LPIPS with the AlexNet variant; PSNR is
capped at 60 dB. The damaged-region PSNR uses a mask given with `--mask` (white = damaged) or, for inputs
degraded with the synthetic noise used in the paper, a mask recovered from the input.

## Citation

If you use this code, please cite the paper above (the full reference will be added on publication).

## License

MIT License, see `LICENSE`.
