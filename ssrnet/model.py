"""SSR-Net: Spatial-attention and Soft-mask Restoration Network.

A U-Net encoder-decoder with spatial attention after the first encoder stage, a reference encoder and an
8x8 cross-attention bottleneck, and two heads on a shared feature: a restoration head producing a candidate
image G and a damage head producing mask logits M. The restored image is obtained by soft-mask fusion

    I_out = I_in + sigmoid(M) * (G - I_in),

so that pixels predicted as intact are passed through from the input. Images are RGB tensors in [-1, 1].
"""
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class ConvDown(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvUp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(in_c, out_c, 3, 1, 1), nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class RefEncoder(nn.Module):
    """Encoder for the reference character (same depth as the main encoder)."""

    def __init__(self):
        super().__init__()
        self.e1 = ConvDown(3, 64, norm=False)
        self.e2 = ConvDown(64, 128)
        self.e3 = ConvDown(128, 256)
        self.e4 = ConvDown(256, 512)
        self.e5 = ConvDown(512, 512)

    def forward(self, ref):
        return self.e5(self.e4(self.e3(self.e2(self.e1(ref)))))


class CrossAttentionBottleneck(nn.Module):
    """Cross-attention at 8x8 resolution: queries from the main branch, keys/values from the reference."""

    def __init__(self, dim=512, heads=8, ff_mult=2):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))

    def forward(self, x, ref):
        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        kv = ref.flatten(2).transpose(1, 2)
        kvn = self.norm_kv(kv)
        a, _ = self.attn(self.norm_q(q), kvn, kvn, need_weights=False)
        z = q + a
        z = z + self.ff(self.norm_ff(z))
        return z.transpose(1, 2).reshape(b, c, h, w)


class SSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = ConvDown(6, 64, norm=False)  # damaged RGB + reference RGB
        self.att1 = SpatialAttention()
        self.d2 = ConvDown(64, 128)
        self.d3 = ConvDown(128, 256)
        self.d4 = ConvDown(256, 512)
        self.d5 = ConvDown(512, 512)
        self.ref_encoder = RefEncoder()
        self.cross_attn = CrossAttentionBottleneck(dim=512, heads=8)
        self.u1 = ConvUp(512, 512)
        self.u2 = ConvUp(1024, 256)
        self.u3 = ConvUp(512, 128)
        self.u4 = ConvUp(256, 64)
        self.shared = nn.Sequential(nn.Conv2d(128, 96, 3, 1, 1), nn.InstanceNorm2d(96), nn.ReLU(inplace=True))
        self.restoration_head = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                              nn.Conv2d(96, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                              nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())
        self.mask_head = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                       nn.Conv2d(96, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x, ref):
        """Returns the candidate image G and the damage-mask logits M."""
        d1 = self.att1(self.d1(torch.cat([x, ref], dim=1)))
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        b = self.cross_attn(d5, self.ref_encoder(ref))
        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1, d4], dim=1))
        u3 = self.u3(torch.cat([u2, d3], dim=1))
        u4 = self.u4(torch.cat([u3, d2], dim=1))
        h = self.shared(torch.cat([u4, d1], dim=1))
        return self.restoration_head(h), self.mask_head(h)


def soft_fusion(x, generated, mask_logits):
    """I_out = I_in + sigmoid(M) * (G - I_in); returns the fused image and the damage probability map."""
    prob = torch.sigmoid(mask_logits)
    return x + prob * (generated - x), prob


@torch.no_grad()
def restore(model, x, ref):
    """Inference: soft-mask fusion of the candidate image and the input."""
    generated, logits = model(x, ref)
    return soft_fusion(x, generated, logits)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = SSRNet().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()
