"""Restore damaged character images with a trained SSR-Net.

Each input is restored together with a reference character (an intact character image, preferably from the
same stele). --ref may be a single image used for all inputs, or a folder: a file with the same name as the
input is used if present, otherwise a reference is drawn at random from the folder (fixed by --seed).
Optionally a CSV with columns target,reference fixes the pairing.

Inputs are resized to 256x256. The restored images are written as PNG; with --save-mask the predicted damage
probability map is written as well (white = reconstructed).

Usage
  python test.py --weights ssrnet_weights.pth --input damaged/ --ref references/ --out restored/ --save-mask
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from PIL import Image

from ssrnet.data import IMG_SIZE, is_img
from ssrnet.model import load_model, restore


def to_tensor(path):
    rgb = np.asarray(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1) / 127.5 - 1.0)[None]


def to_png(t, path):
    rgb = ((t[0].clamp(-1, 1).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).round().astype(np.uint8)
    Image.fromarray(rgb).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help="damaged image or folder of images")
    ap.add_argument("--ref", required=True, help="reference image or folder of reference images")
    ap.add_argument("--pairs", help="optional CSV with columns target,reference (file names)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-mask", action="store_true", help="also save the predicted damage map")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    inputs = ([args.input] if os.path.isfile(args.input) else
              [os.path.join(args.input, f) for f in sorted(os.listdir(args.input)) if is_img(f)])
    refs = [args.ref] if os.path.isfile(args.ref) else sorted(f for f in os.listdir(args.ref) if is_img(f))
    pairs = {r["target"]: r["reference"] for r in csv.DictReader(open(args.pairs, encoding="utf-8"))} if args.pairs else {}
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.weights, args.device)
    for p in inputs:
        name = os.path.basename(p)
        if os.path.isfile(args.ref):
            ref = args.ref
        else:
            ref = os.path.join(args.ref, pairs.get(name) or (name if name in refs else rng.choice(refs)))
        out, prob = restore(model, to_tensor(p).to(args.device), to_tensor(ref).to(args.device))
        stem = os.path.splitext(name)[0]
        to_png(out, os.path.join(args.out, stem + ".png"))
        if args.save_mask:
            Image.fromarray((prob[0, 0].cpu().numpy() * 255).round().astype(np.uint8)).save(
                os.path.join(args.out, stem + "_damage.png"))
    print(f"restored {len(inputs)} image(s) -> {args.out}")


if __name__ == "__main__":
    main()
