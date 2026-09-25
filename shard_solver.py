#!/usr/bin/env python3
"""
Shatter a window, then have a neural net put it back together.

Every training step takes a window, breaks it along fresh random crack lines
into 10-16 shards, spins each shard to a random angle and shuffles them.
The network sees only the loose shards and must say, for each one:
  * where its centre belongs in the window (x, y)
  * how far it has been rotated

A transformer lets the shards "look at each other" (the gold bit that must be
the halo sits above the pink bit that must be the face...).  Finally every
shard is snapped to the nearest empty hole with the Hungarian algorithm, so
no two shards claim the same place.

Usage
  python shard_solver.py train --images data/synthetic --minutes 30
  python shard_solver.py demo  --image some_window.jpg --out result.png
  python shard_solver.py eval  --images data/heldout
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from shards import break_image, render_shards, snap_to_lead

SIZE = 288      # window is resized to SIZE x SIZE
CROP = 104      # shard crop at full res
TILE = 48       # what the network sees
KMIN, KMAX = 24, 48


def load_image(path, size=None):
    size = size or SIZE
    im = Image.open(path).convert("RGB")
    return np.array(im.resize((size, size), Image.BICUBIC))


def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    return sorted(p for e in exts for p in glob.glob(os.path.join(folder, e)))


# ───────────────────────── data ─────────────────────────

def crack_pool(n=2000, path=None):
    """Crack layouts don't depend on the picture, so make a pool once and reuse it
    (with 8 flips/transposes each). Drawing them fresh every step was the bottleneck."""
    path = path or f"checkpoints/cracks_{SIZE}_{KMIN}-{KMAX}.npy"
    if os.path.exists(path):
        pool = np.load(path)
        if len(pool) >= n:
            return pool
    rng = np.random.default_rng(1234)
    pool = np.stack([break_image(SIZE, SIZE, int(rng.integers(KMIN, KMAX + 1)), rng)
                     for _ in range(n)]).astype(np.int8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, pool)
    return pool


def random_cracks(pool, rng):
    lab = pool[rng.integers(len(pool))]
    if rng.random() < 0.5:
        lab = lab[::-1]
    if rng.random() < 0.5:
        lab = lab[:, ::-1]
    if rng.random() < 0.5:
        lab = lab.T
    return np.ascontiguousarray(lab).astype(np.int64)


class ShatterDataset(Dataset):
    """Every access shatters a random window along a random crack pattern."""

    def __init__(self, paths, samples_per_epoch=4000, augment=True, lead=0.0):
        self.imgs = [load_image(p) for p in paths]
        self.pool = crack_pool()
        self.crop = CROP  # kept here: on Windows, workers re-import the module defaults
        self.lead = lead  # fraction of breaks whose cracks are moved onto the lead lines
        self.n = samples_per_epoch
        self.augment = augment

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img = self.imgs[rng.integers(len(self.imgs))]
        if self.augment:
            if rng.random() < 0.5:
                img = img[:, ::-1]
            img = np.clip(img * rng.uniform(0.8, 1.2, (1, 1, 3)), 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        labels = random_cracks(self.pool, rng)
        if rng.random() < self.lead:
            labels = snap_to_lead(img, labels, rng=rng)
        s = render_shards(img, labels, self.crop, TILE, rng=rng)
        return s["tiles"], s["centers"], s["angles"]


def collate(batch):
    B = len(batch)
    K = max(b[0].shape[0] for b in batch)
    tiles = torch.zeros(B, K, 4, TILE, TILE)
    centers = torch.zeros(B, K, 2)
    angles = torch.zeros(B, K)
    pad = torch.ones(B, K, dtype=torch.bool)
    for j, (t, c, a) in enumerate(batch):
        k = t.shape[0]
        tiles[j, :k] = torch.from_numpy(t)
        centers[j, :k] = torch.from_numpy(c)
        angles[j, :k] = torch.from_numpy(a)
        pad[j, :k] = False
    return tiles, centers, angles, pad


# ───────────────────────── model ─────────────────────────

class ShardEncoder(nn.Module):
    def __init__(self, d=192):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                 nn.MaxPool2d(2))
        self.net = nn.Sequential(block(4, 32), block(32, 64), block(64, 128), block(128, d))
        self.proj = nn.Linear(d * 2, d)

    def forward(self, x):
        f = self.net(x)                                  # (N, d, 3, 3)
        m = x[:, 3:4]                                    # mask-weighted pooling
        m = F.adaptive_avg_pool2d(m, f.shape[-1]) + 1e-4
        avg = (f * m).sum((2, 3)) / m.sum((2, 3))
        mx = f.amax((2, 3))
        return self.proj(torch.cat([avg, mx], 1))


class ShardSolver(nn.Module):
    def __init__(self, d=192, layers=4, heads=6):
        super().__init__()
        self.enc = ShardEncoder(d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, layers)
        self.pos = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.rot = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, tiles, pad):
        B, K = tiles.shape[:2]
        f = self.enc(tiles.flatten(0, 1)).view(B, K, -1)
        f = self.mix(f, src_key_padding_mask=pad)
        xy = torch.sigmoid(self.pos(f))
        cs = F.normalize(self.rot(f), dim=-1)            # (cos, sin)
        return xy, cs


def losses(xy, cs, centers, angles, pad):
    valid = ~pad
    l_pos = F.smooth_l1_loss(xy[valid], centers[valid], beta=0.05)
    tgt = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
    l_rot = (1 - (cs * tgt).sum(-1))[valid].mean()
    return l_pos, l_rot


# ───────────────────────── solving ─────────────────────────

def assign(pred_xy, hole_xy):
    """Hungarian: one shard per hole, minimising total distance."""
    cost = np.linalg.norm(pred_xy[:, None] - hole_xy[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    out = np.empty(len(pred_xy), int)
    out[r] = c
    return out


def angle_err_deg(cs, angles):
    pred = np.arctan2(cs[:, 1], cs[:, 0])
    d = np.abs((pred - angles + np.pi) % (2 * np.pi) - np.pi)
    return np.degrees(d)


@torch.no_grad()
def evaluate_model(model, imgs, trials=3, seed=0, lead=0.0):
    model.eval()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    ok = tot = perfect = runs = 0
    rot_errs = []
    for img in imgs:
        for _ in range(trials):
            k = int(rng.integers(KMIN, KMAX + 1))
            labels = break_image(SIZE, SIZE, k, rng)
            if rng.random() < lead:
                labels = snap_to_lead(img, labels, rng=rng)
            s = render_shards(img, labels, CROP, TILE, rng=rng)
            xy, cs = model(torch.from_numpy(s["tiles"])[None].to(dev),
                           torch.zeros(1, len(s["tiles"]), dtype=torch.bool, device=dev))
            a = assign(xy[0].cpu().numpy(), s["centers"])
            hits = (a == np.arange(len(a)))
            ok += hits.sum(); tot += len(a); perfect += hits.all(); runs += 1
            rot_errs += list(angle_err_deg(cs[0].cpu().numpy(), s["angles"]))
    rot_errs = np.array(rot_errs)
    return dict(shard_acc=ok / tot, perfect=perfect / runs,
                rot_median=float(np.median(rot_errs)),
                rot_within_15=float((rot_errs < 15).mean()))


# ───────────────────────── train ─────────────────────────

def device(a):
    if a.device != "auto":
        return torch.device(a.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(a):
    torch.set_num_threads(os.cpu_count())
    dev = device(a)
    paths = list_images(a.images)
    # hold out whole source windows: panes from split_panes.py are named <window>__NN.jpg
    groups = sorted({os.path.basename(p).split("__")[0] for p in paths})
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    val_groups = set(groups[:min(max(4, len(groups) // 20), len(groups) // 2)])
    val_paths = [p for p in paths if os.path.basename(p).split("__")[0] in val_groups]
    tr_paths = [p for p in paths if os.path.basename(p).split("__")[0] not in val_groups]
    rng.shuffle(val_paths)
    print(f"{len(tr_paths)} training images, {len(val_paths)} held-out images, on {dev}")

    ds = ShatterDataset(tr_paths, a.samples, lead=a.lead)
    dl = DataLoader(ds, batch_size=a.batch, collate_fn=collate,
                    num_workers=a.workers, persistent_workers=a.workers > 0,
                    pin_memory=dev.type == "cuda")
    val_imgs = [load_image(p) for p in val_paths[:40]]

    model = ShardSolver().to(dev)
    if a.resume and os.path.exists(a.ckpt):
        model.load_state_dict(torch.load(a.ckpt, weights_only=True, map_location=dev))
        print("resumed from", a.ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    total_steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total_steps,
                                                pct_start=0.05)
    t0 = time.time()
    best = -1
    if a.resume and os.path.exists(a.ckpt):
        # don't let a weak early epoch overwrite the model we resumed from
        best = evaluate_model(model, val_imgs, trials=2, seed=0, lead=a.lead)["shard_acc"]
        print(f"starting point: held-out shards placed {best:.1%}")
    top = []   # (score, epoch, path) of the a.keep best epochs so far
    stem = os.path.splitext(a.ckpt)[0]
    if best >= 0 and a.keep:
        # the starting model competes as "epoch 0", so it is only replaced if beaten
        torch.save(model.state_dict(), f"{stem}_ep000.pt")
        top.append((best, 0, f"{stem}_ep000.pt"))
    step = 0
    for ep in range(1, a.epochs + 1):
        model.train()
        lp = lr_ = 0
        for tiles, centers, angles, pad in dl:
            tiles, centers, angles, pad = (t.to(dev, non_blocking=True)
                                           for t in (tiles, centers, angles, pad))
            xy, cs = model(tiles, pad)
            l_pos, l_rot = losses(xy, cs, centers, angles, pad)
            loss = 10 * l_pos + l_rot
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total_steps - 1:
                sched.step()
            step += 1
            lp += l_pos.item(); lr_ += l_rot.item()
        m = evaluate_model(model, val_imgs, trials=2, seed=ep, lead=a.lead)
        mins = (time.time() - t0) / 60
        print(f"ep {ep:3d} [{mins:5.1f} min] pos {lp / len(dl):.4f} rot {lr_ / len(dl):.3f} | "
              f"held-out: shards placed {m['shard_acc']:.1%}, perfect windows {m['perfect']:.0%}, "
              f"rot median {m['rot_median']:.0f}°", flush=True)
        if m["shard_acc"] > best:
            best = m["shard_acc"]
            torch.save(model.state_dict(), a.ckpt)
        if a.keep and (len(top) < a.keep or m["shard_acc"] > top[-1][0]):
            path = f"{stem}_ep{ep:03d}.pt"
            torch.save(model.state_dict(), path)
            top.append((m["shard_acc"], ep, path))
            top.sort(key=lambda t: -t[0])
            for _, _, old in top[a.keep:]:
                os.remove(old)
            top = top[:a.keep]
        if a.minutes and mins > a.minutes:
            print("time budget reached")
            break
    print(f"best held-out shard accuracy {best:.1%}, saved {a.ckpt}")
    if len(top) > 1:
        # the per-epoch score uses few shatterings and is noisy; re-test the
        # top epochs on every held-out image, more times, and keep the real best
        imgs = [load_image(p) for p in val_paths]
        print(f"re-testing the top {len(top)} epochs on {len(imgs)} held-out images "
              f"x {a.final_trials} shatterings:")
        scores = []
        for _, ep, path in sorted(top, key=lambda t: t[1]):
            model.load_state_dict(torch.load(path, weights_only=True, map_location=dev))
            m = evaluate_model(model, imgs, trials=a.final_trials, seed=12345, lead=a.lead)
            scores.append((m["shard_acc"], ep, path))
            print(f"  ep {ep:3d}: shards placed {m['shard_acc']:.1%}, perfect windows "
                  f"{m['perfect']:.1%}, rot median {m['rot_median']:.0f}°  ({path})", flush=True)
        acc, ep, path = max(scores)
        shutil.copyfile(path, a.ckpt)
        print(f"kept epoch {ep} ({acc:.1%}) as {a.ckpt}")


# ───────────────────────── demo ─────────────────────────

def paste_rgba(canvas, rgba, cx, cy, angle_deg=0.0):
    im = Image.fromarray(rgba, "RGBA")
    if angle_deg:
        im = im.rotate(angle_deg, resample=Image.BICUBIC)
    canvas.alpha_composite(im, (int(round(cx - im.width / 2)), int(round(cy - im.height / 2))))


@torch.no_grad()
def demo(a):
    model = ShardSolver()
    model.load_state_dict(torch.load(a.ckpt, weights_only=True, map_location="cpu"))
    model.eval()  # one window: CPU is plenty
    rng = np.random.default_rng(a.seed)
    img = load_image(a.image)
    k = a.pieces
    labels = break_image(SIZE, SIZE, k, rng)
    s = render_shards(img, labels, CROP, TILE, rng=rng)
    K = len(s["tiles"])
    xy, cs = model(torch.from_numpy(s["tiles"])[None], torch.zeros(1, K, dtype=torch.bool))
    xy, cs = xy[0].numpy(), cs[0].numpy()
    slot = assign(xy, s["centers"])
    pred_ang = np.arctan2(cs[:, 1], cs[:, 0])

    bg = (18, 18, 22, 255)
    P = SIZE + 40
    # 1) original with crack lines
    e = np.zeros(labels.shape, bool)
    e[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    e[1:] |= labels[1:] != labels[:-1]
    orig = img.copy(); orig[e] = (240, 240, 240)
    p1 = Image.new("RGBA", (P, P), bg)
    p1.paste(Image.fromarray(orig), (20, 20))
    # 2) scattered shards
    p2 = Image.new("RGBA", (P, P), bg)
    order = rng.permutation(K)
    cols = int(math.ceil(math.sqrt(K)))
    for n, i in enumerate(order):
        gx, gy = n % cols, n // cols
        cx = 20 + (gx + 0.5) * SIZE / cols + rng.uniform(-6, 6)
        cy = 20 + (gy + 0.5) * SIZE / cols + rng.uniform(-6, 6)
        sm = Image.fromarray(s["rgba"][i], "RGBA").resize((CROP * 2 // 3, CROP * 2 // 3))
        paste_rgba(p2, np.array(sm), cx, cy)
    # 3) reassembled: shard i goes to hole slot[i], un-rotated by predicted angle
    p3 = Image.new("RGBA", (P, P), bg)
    d = ImageDraw.Draw(p3)
    d.rectangle([20, 20, 20 + SIZE, 20 + SIZE], fill=(40, 40, 46, 255))
    correct = 0
    for i in range(K):
        hx, hy = s["centers"][slot[i]] * SIZE
        paste_rgba(p3, s["rgba"][i], 20 + hx, 20 + hy, -np.degrees(pred_ang[i]))
        correct += slot[i] == i
    # outline wrongly placed shards in red
    wrong = np.zeros(labels.shape, bool)
    for i in range(K):
        if slot[i] != i:
            m = labels == slot[i]
            b = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            wrong |= b
    arr = np.array(p3)
    arr[20:20 + SIZE, 20:20 + SIZE][wrong] = (255, 60, 60, 255)
    p3 = Image.fromarray(arr)

    out = Image.new("RGBA", (P * 3, P), bg)
    for j, p in enumerate([p1, p2, p3]):
        out.paste(p, (j * P, 0))
    out = out.convert("RGB").resize((P * 3 * 2, P * 2), Image.LANCZOS)
    out.save(a.out)
    rot = angle_err_deg(cs, s["angles"])
    print(f"{a.image}: {correct}/{K} shards in the right place, "
          f"median rotation error {np.median(rot):.0f}°  → {a.out}")


def evaluate(a):
    dev = device(a)
    model = ShardSolver().to(dev)
    model.load_state_dict(torch.load(a.ckpt, weights_only=True, map_location=dev))
    imgs = [load_image(p) for p in list_images(a.images)[:a.n]]
    m = evaluate_model(model, imgs, trials=a.trials, lead=a.lead)
    print(f"{len(imgs)} windows x {a.trials} shatterings")
    print(f"  shards in correct place : {m['shard_acc']:.1%}")
    print(f"  fully correct windows   : {m['perfect']:.1%}")
    print(f"  median rotation error   : {m['rot_median']:.1f}°  "
          f"({m['rot_within_15']:.0%} within 15°)")


def main():
    global SIZE, KMIN, KMAX, CROP
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--images", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--minutes", type=float, default=0)
    t.add_argument("--samples", type=int, default=3000, help="shatterings per epoch")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--keep", type=int, default=3, help="also keep this many best epochs")
    t.add_argument("--final-trials", type=int, default=5,
                   help="shatterings per image when re-testing the kept epochs at the end")
    d = sub.add_parser("demo")
    d.add_argument("--image", required=True)
    d.add_argument("--out", default="result.png")
    d.add_argument("--pieces", type=int, default=12)
    d.add_argument("--seed", type=int, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--images", required=True)
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--trials", type=int, default=3)
    for p in (t, e):
        p.add_argument("--lead", type=float, default=0.0,
                       help="fraction of breaks whose cracks follow the lead lines (0-1)")
    for p in (t, d, e):
        p.add_argument("--ckpt", default="checkpoints/shards.pt")
        p.add_argument("--size", type=int, default=SIZE, help="window is resized to this")
        p.add_argument("--kmin", type=int, default=KMIN, help="fewest shards")
        p.add_argument("--kmax", type=int, default=KMAX, help="most shards")
        p.add_argument("--crop", type=int, default=CROP, help="shard crop size at full res")
        p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    a = ap.parse_args()
    SIZE, KMIN, KMAX, CROP = a.size, a.kmin, a.kmax, a.crop
    os.makedirs("checkpoints", exist_ok=True)
    {"train": train, "demo": demo, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
