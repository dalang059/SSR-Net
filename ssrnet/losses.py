"""Loss functions, the multi-scale discriminator (stage 1) and the EMA of the generator weights."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .data import denorm01


def masked_mean(loss_map, mask, eps=1e-6):
    m = mask.expand(-1, loss_map.shape[1], -1, -1) if loss_map.shape[1] != 1 else mask
    return (loss_map * m).sum() / (m.sum() + eps)


class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG-19 features (layers [:30], ImageNet normalization)."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:30].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def prep(self, x):
        return (denorm01(x) - self.mean) / self.std

    def forward(self, x, y):
        return F.l1_loss(self.vgg(self.prep(x)), self.vgg(self.prep(y)))


class SobelEdgeLoss(nn.Module):
    """L1 distance between Sobel gradient magnitudes, inside the damage mask dilated by 5x5."""

    def __init__(self):
        super().__init__()
        self.register_buffer("kx", torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer("ky", torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def edge(self, x):
        g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(g, self.kx, padding=1)
        gy = F.conv2d(g, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, gt, mask):
        m = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
        return masked_mean(torch.abs(self.edge(pred) - self.edge(gt)), m)


def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def mask_bce(logits, target):
    """Binary cross-entropy with the positive class weighted by neg/pos (clamped to [1, 20])."""
    pos = target.sum()
    pos_weight = ((target.numel() - pos) / (pos + 1e-6)).clamp(min=1.0, max=20.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*block(6, 64, norm=False), *block(64, 128), *block(128, 256), nn.Conv2d(256, 1, 4, 1, 1))

    def forward(self, x):
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    """Conditional PatchGAN at full and half resolution; input = (degraded, restored or ground truth)."""

    def __init__(self):
        super().__init__()
        self.local = NLayerDiscriminator()
        self.global_d = NLayerDiscriminator()
        self.down = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        return self.local(z), self.global_d(self.down(z))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        s = dict(self.shadow.named_parameters())
        for name, p in model.named_parameters():
            s[name].mul_(self.decay).add_(p, alpha=1.0 - self.decay)
        sb = dict(self.shadow.named_buffers())
        for name, b in model.named_buffers():
            sb[name].copy_(b)
