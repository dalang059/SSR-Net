"""Train SSR-Net.

The model reported in the paper was trained in two stages:
  stage 1  from random initialization, with a multi-scale LSGAN discriminator (configs/stage1.json)
  stage 2  fine-tuning of the selected stage-1 weights without the discriminator (configs/stage2.json)
In both stages an exponential moving average (EMA) of the weights is evaluated on a character-level
validation split after every epoch, and the EMA weights with the highest validation PSNR are kept.

Usage
  python train.py --stage 1 --data /path/to/pairs --out runs/stage1
  python train.py --stage 2 --data /path/to/pairs --out runs/stage2 --init runs/stage1/best_ema.pth
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from ssrnet.data import PairDataset, base_key, denorm01, grouped_split, reference_pool
from ssrnet.losses import (EMA, MultiscaleDiscriminator, SobelEdgeLoss, VGGPerceptualLoss, dice_loss_from_logits,
                           mask_bce, masked_mean)
from ssrnet.model import SSRNet, soft_fusion


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batch_psnr(pred01, gt01):
    return 10.0 * torch.log10(1.0 / (((pred01 - gt01) ** 2).mean(dim=(1, 2, 3)) + 1e-12))


def output_of(mode, x, generated, logits):
    """'fused': soft-mask fusion I_in + sigmoid(M)(G - I_in); 'direct': the candidate image G."""
    return soft_fusion(x, generated, logits)[0] if mode == "fused" else generated


@torch.no_grad()
def validate(model, loader, mode, device):
    model.eval()
    psnr, ss, iou = [], [], []
    for x, ref, gt_mask, gt, _ in loader:
        x, ref, gt_mask, gt = x.to(device), ref.to(device), gt_mask.to(device), gt.to(device)
        generated, logits = model(x, ref)
        p01, g01 = denorm01(output_of(mode, x, generated, logits)), denorm01(gt)
        psnr += batch_psnr(p01, g01).cpu().tolist()
        ss += ssim(p01, g01, data_range=1.0, size_average=False).cpu().tolist()
        pm = (torch.sigmoid(logits) >= 0.5).float()
        inter = (pm * gt_mask).sum(dim=(1, 2, 3))
        union = ((pm + gt_mask) > 0).float().sum(dim=(1, 2, 3))
        iou += ((inter + 1e-6) / (union + 1e-6)).cpu().tolist()
    return float(np.mean(psnr)), float(np.mean(ss)), float(np.mean(iou))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--data", required=True, help="folder with Input/ and Target/ subfolders of paired images")
    ap.add_argument("--out", required=True, help="output folder for checkpoints and logs")
    ap.add_argument("--config", help="configuration file (default: configs/stage<N>.json)")
    ap.add_argument("--init", help="checkpoint to start from (required for stage 2)")
    ap.add_argument("--epochs", type=int, help="override the number of epochs in the configuration")
    ap.add_argument("--max-batches", type=int, help="debugging only: stop each epoch after this many batches")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(args.config or os.path.join(here, "configs", f"stage{args.stage}.json"), encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.stage == 2 and not args.init:
        ap.error("stage 2 needs --init (the best stage-1 checkpoint)")
    w = cfg["loss_weights"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])
    os.makedirs(args.out, exist_ok=True)
    json.dump({**cfg, "stage": args.stage, "init": args.init}, open(os.path.join(args.out, "config_used.json"), "w"), indent=2)

    # data: character-level split; references only from training characters of the same stele
    files, train_files, val_files, val_groups = grouped_split(args.data, cfg["val_ratio"], cfg["seed"])
    pool = reference_pool(train_files)
    with open(os.path.join(args.out, "split_manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "base_group", "split"])
        for f in files:
            wr.writerow([f, base_key(f), "validation" if base_key(f) in val_groups else "training"])
    print(f"pairs: {len(files)}; training {len(train_files)}, validation {len(val_files)}; device {device}")
    kw = dict(seed=cfg["seed"], mask_threshold=cfg["mask_diff_threshold"], mask_dilate=cfg["mask_dilate"])
    train_loader = DataLoader(PairDataset(args.data, train_files, pool, training=True, **kw), batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(PairDataset(args.data, val_files, pool, training=False, **kw), batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    # model, optimizers, schedules
    net_G = SSRNet().to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        net_G.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
        print("initialized from", args.init)
    Opt = torch.optim.Adam if cfg["optimizer"] == "adam" else torch.optim.AdamW
    opt_G = Opt(net_G.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    E, W = cfg["epochs"], cfg["warmup_epochs"]

    def lr_factor(epoch):  # linear warm-up, then cosine decay
        if epoch < W:
            return (epoch + 1) / W
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - W) / max(1, E - W - 1)))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_factor)
    if cfg["adversarial"]:
        net_D = MultiscaleDiscriminator().to(device)
        opt_D = torch.optim.Adam(net_D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
        sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lr_factor)
    vgg_loss, edge_loss = VGGPerceptualLoss().to(device), SobelEdgeLoss().to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(net_G, cfg["ema_decay"])

    best_path, last_path = os.path.join(args.out, "best_ema.pth"), os.path.join(args.out, "last.pth")
    log = open(os.path.join(args.out, "log.csv"), "w", newline="")
    logw = csv.writer(log)
    logw.writerow(["epoch", "train_loss_G", "val_psnr", "val_ssim", "val_mask_iou", "lr", "minutes"])

    def save_best(epoch, v):
        torch.save({"epoch": epoch, "model": ema.shadow.state_dict(), "val_psnr": v[0], "val_ssim": v[1],
                    "val_mask_iou": v[2], "stage": args.stage, "seed": cfg["seed"]}, best_path)

    best, best_epoch, stale = -1e9, -1, 0
    if cfg["validate_init"]:  # the starting weights remain the fallback if fine-tuning does not help
        v = validate(ema.shadow, val_loader, cfg["val_output"], device)
        best, best_epoch = v[0], 0
        save_best(0, v)
        print(f"initial validation: PSNR {v[0]:.3f}, SSIM {v[1]:.5f}, mask IoU {v[2]:.4f}")

    for epoch in range(E):
        t0, run, nb = time.time(), 0.0, 0
        net_G.train()
        if cfg["adversarial"]:
            net_D.train()
        for i, (x, ref, gt_mask, gt, _) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            x, ref, gt_mask, gt = x.to(device), ref.to(device), gt_mask.to(device), gt.to(device)
            opt_G.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                generated, logits = net_G(x, ref)
                out = output_of(cfg["train_output"], x, generated, logits)
                p01, g01 = denorm01(out), denorm01(gt)
                loss = (w["full_l1"] * F.l1_loss(out, gt)
                        + w["mask_l1"] * masked_mean(torch.abs(out - gt), gt_mask)
                        + w["mask_mse"] * masked_mean((out - gt) ** 2, gt_mask)
                        + w["ssim"] * (1.0 - ssim(p01, g01, data_range=1.0, size_average=True))
                        + w["vgg"] * vgg_loss(out, gt)
                        + w["edge"] * edge_loss(out, gt, gt_mask)
                        + w["mask_bce"] * mask_bce(logits, gt_mask)
                        + w["mask_dice"] * dice_loss_from_logits(logits, gt_mask))
                if w["intact"]:
                    loss = loss + w["intact"] * masked_mean(torch.abs(out - x), 1.0 - gt_mask)
                if cfg["adversarial"]:
                    fl, fg = net_D(x, out)
                    loss = loss + w["gan"] * 0.5 * (F.mse_loss(fl, torch.ones_like(fl)) + F.mse_loss(fg, torch.ones_like(fg)))
            scaler.scale(loss).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(net_G.parameters(), max_norm=cfg["grad_clip"])
            scaler.step(opt_G)
            if cfg["adversarial"]:  # least-squares GAN discriminator update
                opt_D.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    rl, rg = net_D(x, gt)
                    fl, fg = net_D(x, out.detach())
                    loss_D = 0.5 * (0.5 * (F.mse_loss(rl, torch.ones_like(rl)) + F.mse_loss(rg, torch.ones_like(rg)))
                                    + 0.5 * (F.mse_loss(fl, torch.zeros_like(fl)) + F.mse_loss(fg, torch.zeros_like(fg))))
                scaler.scale(loss_D).backward()
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(net_D.parameters(), max_norm=cfg["grad_clip"])
                scaler.step(opt_D)
            scaler.update()
            ema.update(net_G)
            run, nb = run + loss.item(), nb + 1
            if i % 200 == 0:
                print(f"[epoch {epoch + 1}/{E}] [batch {i}/{len(train_loader)}] loss_G {loss.item():.3f}", flush=True)
        sched_G.step()
        if cfg["adversarial"]:
            sched_D.step()

        v = validate(ema.shadow, val_loader, cfg["val_output"], device)
        minutes = (time.time() - t0) / 60
        logw.writerow([epoch + 1, f"{run / max(1, nb):.4f}", f"{v[0]:.4f}", f"{v[1]:.5f}", f"{v[2]:.4f}",
                       f"{opt_G.param_groups[0]['lr']:.2e}", f"{minutes:.1f}"])
        log.flush()
        print(f"epoch {epoch + 1}: val PSNR {v[0]:.3f}, SSIM {v[1]:.5f}, mask IoU {v[2]:.4f} ({minutes:.1f} min)", flush=True)
        if v[0] > best + 1e-6:
            best, best_epoch, stale = v[0], epoch + 1, 0
            save_best(epoch + 1, v)
            print(f"  new best (epoch {best_epoch})")
        else:
            stale += 1
        state = {"epoch": epoch + 1, "net_G": net_G.state_dict(), "ema": ema.shadow.state_dict(), "opt_G": opt_G.state_dict(),
                 "best_val_psnr": best, "best_epoch": best_epoch}
        if cfg["adversarial"]:
            state.update(net_D=net_D.state_dict(), opt_D=opt_D.state_dict())
        torch.save(state, last_path)
        if cfg["patience"] and stale >= cfg["patience"]:
            print(f"early stopping at epoch {epoch + 1}")
            break
    print(f"done. best validation PSNR {best:.3f} at epoch {best_epoch}; weights: {best_path}")


if __name__ == "__main__":
    main()
