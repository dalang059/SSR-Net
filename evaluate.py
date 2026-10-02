"""Compute PSNR, SSIM, LPIPS and damaged-region PSNR of restored images against ground truth.

PSNR and SSIM are computed with scikit-image on 8-bit RGB images scaled to [0, 1]; LPIPS uses the AlexNet
variant. PSNR is capped at 60 dB (an exact reconstruction would otherwise give an infinite value).
The damaged-region PSNR is computed over a binary damage mask: either given as a folder of mask images
(white = damaged) or, for inputs degraded with the synthetic noise used in this work, recovered from the
degraded input as its non-gray pixels (the clean rubbing images are grayscale).

Usage
  python evaluate.py --pred restored/ --gt ground_truth/ --input damaged/ --csv per_image.csv
  python evaluate.py --pred restored/ --gt ground_truth/ --mask masks/
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PSNR_CAP = 60.0


def read(path, size=256):
    im = Image.open(path).convert("RGB")
    return np.asarray(im if im.size == (size, size) else im.resize((size, size), Image.BICUBIC))


def recover_damage_mask(inp_rgb):
    x = inp_rgb.astype(np.int16)
    return (x.max(2) - x.min(2)) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--input", help="degraded inputs, to recover the damage mask (synthetic test sets)")
    ap.add_argument("--mask", help="folder of binary damage masks with the same file names")
    ap.add_argument("--csv", help="write per-image results to this file")
    args = ap.parse_args()

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).eval()
    rows = []
    for name in sorted(f for f in os.listdir(args.gt) if f.lower().endswith((".png", ".jpg"))):
        g = read(os.path.join(args.gt, name)).astype(np.float32) / 255.0
        p = read(os.path.join(args.pred, os.path.splitext(name)[0] + ".png")).astype(np.float32) / 255.0
        r = {"image": name,
             "psnr": min(peak_signal_noise_ratio(g, p, data_range=1.0), PSNR_CAP),
             "ssim": structural_similarity(g, p, data_range=1.0, channel_axis=-1)}
        with torch.no_grad():
            r["lpips"] = lp(torch.from_numpy(p.transpose(2, 0, 1))[None] * 2 - 1,
                            torch.from_numpy(g.transpose(2, 0, 1))[None] * 2 - 1).item()
        m = None
        if args.mask:
            m = np.asarray(Image.open(os.path.join(args.mask, name)).convert("L").resize((256, 256), Image.NEAREST)) > 127
        elif args.input:
            m = recover_damage_mask(read(os.path.join(args.input, name)))
        if m is not None and m.sum() >= 10:
            mse = ((p - g) ** 2)[m].mean()
            r["masked_psnr"] = min(10 * np.log10(1.0 / max(mse, 1e-10)), PSNR_CAP)
        rows.append(r)
    df = pd.DataFrame(rows)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(f"{len(df)} images")
    print(df.drop(columns="image").mean().round(4).to_string())


if __name__ == "__main__":
    main()
