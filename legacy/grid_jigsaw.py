#!/usr/bin/env python3
"""
Window Jigsaw — scramble an intact window, then train a neural net to
reassemble it.

The idea: given a stained-glass window image, cut it into tiles on a
regular grid, shuffle (and optionally rotate 0/90/180/270) the tiles,
and learn to predict each tile's original (row, col, rotation).

Two problems are solved jointly:
  1. **Permutation** – which grid slot does each tile belong to?
  2. **Rotation** – is the tile flipped 0°, 90°, 180°, or 270°?

The model is a lightweight CNN that encodes each tile independently,
then reasons over the full set to produce per-tile predictions.

Usage
-----
# 1. generate synthetic windows (or put your own photos in data/real/)
python make_windows.py --out data/synthetic --n 2000 --size 192

# 2. train on those windows
python window_jigsaw.py train --images data/synthetic --epochs 40

# 3. demo: scramble & reassemble one image
python window_jigsaw.py demo --image data/synthetic/window_00042.png

# 4. evaluate on a folder
python window_jigsaw.py eval --images data/synthetic --n 50
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# Tile manipulation
# ─────────────────────────────────────────────

def split_tiles(img: np.ndarray, rows: int, cols: int) -> list[np.ndarray]:
    """Split HxWx3 image into rows*cols tiles (list in row-major order)."""
    h, w = img.shape[:2]
    th, tw = h // rows, w // cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tiles.append(img[r * th:(r + 1) * th, c * tw:(c + 1) * tw].copy())
    return tiles


def assemble(tiles: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """Stitch tiles back into a single image."""
    row_strips = []
    for r in range(rows):
        row_strips.append(np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1))
    return np.concatenate(row_strips, axis=0)


ROT_FNS = {
    0: lambda t: t,
    1: lambda t: np.rot90(t, k=1),  # 90° CCW
    2: lambda t: np.rot90(t, k=2),  # 180°
    3: lambda t: np.rot90(t, k=3),  # 270°
}


def scramble(tiles: list[np.ndarray], rotate: bool = True, rng=None):
    """Shuffle tiles and optionally rotate each one.

    Returns (scrambled_tiles, perm, rotations) where
        perm[i] = original slot of tile now at position i
        rotations[i] = k in {0,1,2,3} applied to tile i
    """
    rng = rng or random.Random()
    n = len(tiles)
    perm = list(range(n))
    rng.shuffle(perm)
    rots = [rng.randint(0, 3) if rotate else 0 for _ in range(n)]
    out = [ROT_FNS[rots[i]](tiles[perm[i]]) for i in range(n)]
    return out, perm, rots


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class JigsawDataset(Dataset):
    """Yields (scrambled_tiles_tensor, position_labels, rotation_labels).

    Each sample loads one window image, splits it into a grid, scrambles the
    pieces, and the labels say where each piece originally was and how it
    was rotated.
    """

    def __init__(self, image_dir: str, rows=4, cols=3, tile_size=48,
                 rotate=True, augment=True, limit: int | None = None):
        self.paths = sorted(Path(image_dir).glob("*.png"))
        if limit:
            self.paths = self.paths[:limit]
        self.rows = rows
        self.cols = cols
        self.n_tiles = rows * cols
        self.tile_size = tile_size
        self.rotate = rotate
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # resize so tiles divide evenly
        img = img.resize((self.cols * self.tile_size, self.rows * self.tile_size),
                         Image.BILINEAR)
        arr = np.array(img)

        # optional colour jitter
        if self.augment:
            arr = (arr.astype(np.float32) *
                   np.random.uniform(0.85, 1.15, (1, 1, 3))).clip(0, 255).astype(np.uint8)

        tiles = split_tiles(arr, self.rows, self.cols)
        scrambled, perm, rots = scramble(tiles, self.rotate)

        # to tensor: (n_tiles, 3, th, tw) float32 in [0,1]
        t = np.stack(scrambled).astype(np.float32) / 255.0
        t = torch.from_numpy(t).permute(0, 3, 1, 2)

        pos = torch.tensor(perm, dtype=torch.long)
        rot = torch.tensor(rots, dtype=torch.long)
        return t, pos, rot


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class TileEncoder(nn.Module):
    """Small CNN to encode one tile into a feature vector."""
    def __init__(self, tile_size=48, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class JigsawSolver(nn.Module):
    """
    Encode each tile, concatenate all tile features, and predict
    (position, rotation) for every tile.

    Architecture:
        1. Shared TileEncoder → per-tile features  (n_tiles × feat_dim)
        2. Concatenate all features into one global vector
        3. MLP → per-tile position logits  (n_tiles × n_tiles)
        4. Per-tile rotation head           (n_tiles × 4)
    """

    def __init__(self, n_tiles=12, tile_size=48, feat_dim=128, hidden=512):
        super().__init__()
        self.n_tiles = n_tiles
        self.feat_dim = feat_dim

        self.tile_enc = TileEncoder(tile_size, feat_dim)

        global_dim = n_tiles * feat_dim
        self.pos_head = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_tiles * n_tiles),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, tiles):
        """
        tiles: (B, n_tiles, 3, H, W)
        returns: pos_logits (B, n_tiles, n_tiles), rot_logits (B, n_tiles, 4)
        """
        B, N = tiles.shape[:2]
        flat = tiles.reshape(B * N, *tiles.shape[2:])
        feats = self.tile_enc(flat).reshape(B, N, self.feat_dim)   # (B, N, D)

        # global reasoning over all tiles
        global_feat = feats.reshape(B, -1)                         # (B, N*D)
        pos_logits = self.pos_head(global_feat).reshape(B, N, N)   # (B, N, N)

        # per-tile rotation
        rot_logits = self.rot_head(feats)                          # (B, N, 4)

        return pos_logits, rot_logits


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args):
    rows, cols = args.rows, args.cols
    n_tiles = rows * cols
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = JigsawDataset(args.images, rows, cols, args.tile_size,
                       rotate=args.rotate, limit=args.limit)
    val_split = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - val_split, val_split])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=True)

    model = JigsawSolver(n_tiles, args.tile_size, args.feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, correct_pos, correct_rot, total = [], 0, 0, 0
        for tiles, pos, rot in train_dl:
            tiles, pos, rot = tiles.to(device), pos.to(device), rot.to(device)
            pos_logits, rot_logits = model(tiles)

            loss_pos = F.cross_entropy(pos_logits.reshape(-1, n_tiles), pos.reshape(-1))
            loss_rot = F.cross_entropy(rot_logits.reshape(-1, 4), rot.reshape(-1))
            loss = loss_pos + 0.5 * loss_rot

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            correct_pos += (pos_logits.argmax(-1) == pos).sum().item()
            correct_rot += (rot_logits.argmax(-1) == rot).sum().item()
            total += pos.numel()

        sched.step()

        # validation
        model.eval()
        val_pos_ok, val_rot_ok, val_total = 0, 0, 0
        with torch.no_grad():
            for tiles, pos, rot in val_dl:
                tiles, pos, rot = tiles.to(device), pos.to(device), rot.to(device)
                pl, rl = model(tiles)
                val_pos_ok += (pl.argmax(-1) == pos).sum().item()
                val_rot_ok += (rl.argmax(-1) == rot).sum().item()
                val_total += pos.numel()

        val_acc = val_pos_ok / max(1, val_total)
        print(f"Epoch {epoch:3d}  loss {np.mean(losses):.3f}  "
              f"train pos {correct_pos / total:.1%} rot {correct_rot / total:.1%}  "
              f"val pos {val_acc:.1%} rot {val_rot_ok / max(1, val_total):.1%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best.pt"))

    print(f"\nBest val position accuracy: {best_acc:.1%}")
    print(f"Checkpoint saved to {args.ckpt_dir}/best.pt")


# ─────────────────────────────────────────────
# Demo / visualisation
# ─────────────────────────────────────────────

def demo(args):
    rows, cols = args.rows, args.cols
    n_tiles = rows * cols
    device = "cpu"

    model = JigsawSolver(n_tiles, args.tile_size, args.feat_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best.pt"),
                                     map_location=device, weights_only=True))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    img = img.resize((cols * args.tile_size, rows * args.tile_size), Image.BILINEAR)
    arr = np.array(img)

    tiles = split_tiles(arr, rows, cols)
    scrambled, perm, rots = scramble(tiles, rotate=args.rotate)

    # predict
    t = np.stack(scrambled).astype(np.float32) / 255.0
    t = torch.from_numpy(t).permute(0, 3, 1, 2).unsqueeze(0)
    with torch.no_grad():
        pos_logits, rot_logits = model(t)
    pred_pos = pos_logits[0].argmax(-1).tolist()
    pred_rot = rot_logits[0].argmax(-1).tolist()

    # --- unscramble using predictions ---
    reconstructed = [None] * n_tiles
    for i in range(n_tiles):
        # tile i (in scrambled order) → predicted slot pred_pos[i]
        unrot = ROT_FNS[(4 - pred_rot[i]) % 4](scrambled[i])
        slot = pred_pos[i]
        if reconstructed[slot] is None:
            reconstructed[slot] = unrot
    # fill any collisions with black
    th, tw = args.tile_size, args.tile_size
    for i in range(n_tiles):
        if reconstructed[i] is None:
            reconstructed[i] = np.zeros((th, tw, 3), np.uint8)

    # build side-by-side: original | scrambled | reconstructed
    original = assemble(tiles, rows, cols)
    scrambled_img = assemble(scrambled, rows, cols)
    recon_img = assemble(reconstructed, rows, cols)

    # add thin white dividers
    divider = np.full((original.shape[0], 4, 3), 255, np.uint8)
    out = np.concatenate([original, divider, scrambled_img, divider, recon_img], axis=1)

    out_path = args.out or "demo_result.png"
    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")
    print(f"  Position accuracy: {sum(p == gt for p, gt in zip(pred_pos, perm)) / n_tiles:.0%}")
    if args.rotate:
        print(f"  Rotation accuracy: {sum(p == gt for p, gt in zip(pred_rot, rots)) / n_tiles:.0%}")


# ─────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────

def evaluate(args):
    rows, cols = args.rows, args.cols
    n_tiles = rows * cols
    device = "cpu"

    model = JigsawSolver(n_tiles, args.tile_size, args.feat_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best.pt"),
                                     map_location=device, weights_only=True))
    model.eval()

    ds = JigsawDataset(args.images, rows, cols, args.tile_size,
                       rotate=args.rotate, augment=False, limit=args.n)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    pos_ok = rot_ok = total = 0
    perfect = 0
    for tiles, pos, rot in dl:
        with torch.no_grad():
            pl, rl = model(tiles)
        pp = pl.argmax(-1)
        pr = rl.argmax(-1)
        pos_ok += (pp == pos).sum().item()
        rot_ok += (pr == rot).sum().item()
        perfect += ((pp == pos).all(dim=1)).sum().item()
        total += pos.numel()

    n = len(ds)
    print(f"Evaluated on {n} windows ({n_tiles} tiles each)")
    print(f"  Tile position accuracy:  {pos_ok / total:.1%}")
    print(f"  Tile rotation accuracy:  {rot_ok / total:.1%}")
    print(f"  Fully correct windows:   {perfect}/{n} ({perfect / n:.1%})")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Window Jigsaw solver")
    sub = ap.add_subparsers(dest="cmd")

    # shared args
    for name in ("train", "demo", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--rows", type=int, default=4)
        p.add_argument("--cols", type=int, default=3)
        p.add_argument("--tile-size", type=int, default=48)
        p.add_argument("--feat-dim", type=int, default=128)
        p.add_argument("--ckpt-dir", default="checkpoints")
        p.add_argument("--rotate", action="store_true", default=True)
        p.add_argument("--no-rotate", dest="rotate", action="store_false")

    # train-specific
    tp = sub.choices["train"]
    tp.add_argument("--images", required=True)
    tp.add_argument("--epochs", type=int, default=40)
    tp.add_argument("--batch", type=int, default=32)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--limit", type=int, default=None)

    # demo-specific
    dp = sub.choices["demo"]
    dp.add_argument("--image", required=True)
    dp.add_argument("--out", default=None)

    # eval-specific
    ep = sub.choices["eval"]
    ep.add_argument("--images", required=True)
    ep.add_argument("--n", type=int, default=50)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "demo":
        demo(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
