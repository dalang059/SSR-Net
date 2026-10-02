"""Paired training data, character-level train/validation split and reference selection.

Expected layout of the training set (synthetic degraded/clean pairs with identical file names):
    <root>/Input/<stele>_<character>_<a>_<b>.png    degraded image
    <root>/Target/<stele>_<character>_<a>_<b>.png   clean ground truth
The first underscore-separated field identifies the stele; removing the last two fields gives the source
character, so that all degraded versions of one character fall into the same split.
"""
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 256
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_img(name):
    return name.lower().endswith(IMG_EXT)


def stele_key(fname):
    return fname.split("_")[0]


def base_key(fname):
    """Source character of a synthetic pair: the file name without its last two fields."""
    parts = fname.split("_")
    return "_".join(parts[:-2]) if len(parts) >= 3 else os.path.splitext(fname)[0]


def pil_rgb(path):
    return Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


img_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def denorm01(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def derive_training_mask(img_in, img_gt, threshold=18.0, dilate=1):
    """Damage mask for supervision: mean absolute RGB difference > threshold (0-255), dilated by 3x3 steps."""
    diff = np.mean(np.abs(np.asarray(img_in, np.float32) - np.asarray(img_gt, np.float32)), axis=2)
    mask = torch.from_numpy((diff > threshold).astype(np.float32)).unsqueeze(0)
    if dilate > 0:
        k = 2 * dilate + 1
        mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=dilate).squeeze(0)
    return mask.clamp(0, 1)


def grouped_split(root, val_ratio=0.10, seed=20260913):
    """Split the pairs by source character (all degraded versions of a character stay together)."""
    files = sorted(f for f in os.listdir(os.path.join(root, "Input")) if is_img(f))
    groups = defaultdict(list)
    for f in files:
        groups[base_key(f)].append(f)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    n_val = max(1, int(round(len(names) * val_ratio)))
    val_groups = set(names[:n_val])
    train = [f for f in files if base_key(f) not in val_groups]
    val = [f for f in files if base_key(f) in val_groups]
    return files, train, val, val_groups


def reference_pool(train_files):
    """References are drawn only from training characters of the same stele."""
    pool = defaultdict(list)
    for f in train_files:
        pool[stele_key(f)].append(f)
    return {k: sorted(v) for k, v in pool.items()}


class PairDataset(Dataset):
    """Returns (degraded, reference, damage mask, ground truth, file name)."""

    def __init__(self, root, files, ref_pool, training=True, seed=20260913, mask_threshold=18.0, mask_dilate=1):
        self.input_dir = os.path.join(root, "Input")
        self.target_dir = os.path.join(root, "Target")
        self.files = list(files)
        self.ref_pool = ref_pool
        self.training = training
        self.seed = seed
        self.mask_threshold, self.mask_dilate = mask_threshold, mask_dilate

    def __len__(self):
        return len(self.files)

    def pick_reference(self, fname, idx):
        pool = self.ref_pool[stele_key(fname)]
        candidates = [r for r in pool if base_key(r) != base_key(fname)] or pool  # never the same character
        if self.training:
            return random.choice(candidates)
        return random.Random(self.seed + idx * 1009).choice(candidates)  # fixed validation reference

    def __getitem__(self, idx):
        fname = self.files[idx]
        x_img = pil_rgb(os.path.join(self.input_dir, fname))
        gt_img = pil_rgb(os.path.join(self.target_dir, fname))
        mask = derive_training_mask(x_img, gt_img, self.mask_threshold, self.mask_dilate)
        ref_img = pil_rgb(os.path.join(self.target_dir, self.pick_reference(fname, idx)))
        return img_to_tensor(x_img), img_to_tensor(ref_img), mask, img_to_tensor(gt_img), fname
