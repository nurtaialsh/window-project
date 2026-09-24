"""
Generate synthetic stained-glass lancet windows for testing the pipeline.

Real photos of intact windows (Chartres, Canterbury, York, Fairford, etc.) are
what you actually want to train on - drop them in a folder and point
window_jigsaw.py at it. This script exists so the whole pipeline can be run
and debugged before you have a real dataset.

Each window has the kind of global structure real ones do:
  - pointed (lancet) arch at the top, dark stonework outside it
  - a coloured border band
  - background glass that is mostly blue above, red below
  - a standing figure: haloed head, robe, sometimes a canopy above
  - every piece of glass separated by dark lead lines (Voronoi cells)

Usage:
    python make_windows.py --out data/synthetic --n 2000 --size 192
"""
import argparse
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import cKDTree

PALETTES = {
    "blue":   [(20, 50, 140), (30, 70, 170), (15, 35, 110), (50, 90, 190)],
    "red":    [(160, 25, 30), (190, 40, 35), (130, 20, 25), (200, 70, 50)],
    "gold":   [(210, 170, 50), (230, 190, 80), (190, 140, 30)],
    "green":  [(40, 120, 60), (60, 150, 80), (30, 90, 50)],
    "white":  [(220, 215, 190), (235, 230, 210), (200, 195, 170)],
    "purple": [(100, 40, 120), (130, 60, 150)],
    "flesh":  [(225, 190, 160), (210, 170, 140)],
}


def jitter(c, amt=18):
    return tuple(int(np.clip(v + random.randint(-amt, amt), 0, 255)) for v in c)


def lancet_mask(w, h, draw_scale=1):
    """Pointed arch window shape: True inside the glass."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w / 2
    half = w * 0.45
    spring = h * 0.30  # where the arch springs from
    # pointed arch = intersection of two circles of radius 2*half centred on opposite jambs
    r = 2 * half * 0.85
    left_c = cx + half - r
    right_c = cx - half + r
    in_arch = ((xx - left_c) ** 2 + (yy - spring) ** 2 <= r ** 2) & \
              ((xx - right_c) ** 2 + (yy - spring) ** 2 <= r ** 2)
    in_body = (np.abs(xx - cx) <= half) & (yy >= spring) & (yy <= h * 0.97)
    return (in_arch & (yy < spring)) | in_body


def make_window(size=192, seed=None):
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed if seed is not None else None)
    w = h = size
    S = 4  # draw at 4x then downsample for anti-aliasing
    W, H = w * S, h * S

    # --- colour "design" layer: what each pixel of glass *should* be ---
    design = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(design)

    bg_top = rng.choice(["blue", "blue", "red", "green"])
    bg_bot = rng.choice(["red", "blue", "purple"]) if rng.random() < 0.7 else bg_top
    split = int(H * rng.uniform(0.45, 0.7))
    d.rectangle([0, 0, W, split], fill=PALETTES[bg_top][0])
    d.rectangle([0, split, W, H], fill=PALETTES[bg_bot][0])

    cx = W // 2 + int(rng.uniform(-0.05, 0.05) * W)
    # canopy
    if rng.random() < 0.6:
        cy = int(H * 0.2)
        d.polygon([(cx - W * 0.28, cy + H * 0.08), (cx, cy - H * 0.07),
                   (cx + W * 0.28, cy + H * 0.08)], fill=PALETTES["gold"][0])
        d.rectangle([cx - W * 0.3, cy + H * 0.08, cx + W * 0.3, cy + H * 0.12],
                    fill=PALETTES["white"][0])
    # figure
    head_y = int(H * rng.uniform(0.34, 0.42))
    head_r = int(W * rng.uniform(0.055, 0.075))
    halo_r = int(head_r * 1.8)
    d.ellipse([cx - halo_r, head_y - halo_r, cx + halo_r, head_y + halo_r],
              fill=PALETTES["gold"][rng.randrange(3)])
    robe = rng.choice(["red", "green", "white", "purple", "blue"])
    mantle = rng.choice(["green", "white", "gold", "red", "purple"])
    feet_y = int(H * rng.uniform(0.82, 0.9))
    shoulder = int(W * rng.uniform(0.12, 0.16))
    hem = int(W * rng.uniform(0.2, 0.27))
    d.polygon([(cx - shoulder, head_y + head_r), (cx + shoulder, head_y + head_r),
               (cx + hem, feet_y), (cx - hem, feet_y)], fill=PALETTES[robe][0])
    # mantle over one side
    side = rng.choice([-1, 1])
    d.polygon([(cx, head_y + head_r), (cx + side * shoulder, head_y + head_r),
               (cx + side * hem, feet_y - H * 0.08), (cx, feet_y - H * 0.2)],
              fill=PALETTES[mantle][0])
    d.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r],
              fill=PALETTES["flesh"][0])
    # ground
    d.rectangle([0, feet_y, W, H], fill=PALETTES["green"][rng.randrange(3)])
    # base inscription panel
    if rng.random() < 0.5:
        d.rectangle([W * 0.1, H * 0.9, W * 0.9, H], fill=PALETTES["white"][1])

    design = np.array(design.resize((w, h), Image.NEAREST)).astype(np.int32)

    # --- break into glass pieces with Voronoi cells, one colour per piece ---
    n_pieces = rng.randint(int(size * 0.9), int(size * 1.6))
    pts = np.column_stack([np.random.uniform(0, w, n_pieces),
                           np.random.uniform(0, h, n_pieces)])
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist, idx = cKDTree(pts).query(grid, k=2)
    idx0 = idx[:, 0].reshape(h, w)
    lead = (dist[:, 1] - dist[:, 0]).reshape(h, w) < 1.2  # near a boundary

    out = np.zeros((h, w, 3), np.int32)
    for i in range(n_pieces):
        m = idx0 == i
        if not m.any():
            continue
        base = np.median(design[m], axis=0)
        out[m] = jitter(base, 16)
    # boundaries between design regions are also lead lines
    edge = np.zeros((h, w), bool)
    edge[:, 1:] |= np.any(design[:, 1:] != design[:, :-1], axis=2)
    edge[1:, :] |= np.any(design[1:, :] != design[:-1, :], axis=2)
    lead |= edge

    # per-pixel glass texture (streaks, uneven thickness)
    tex = np.random.normal(0, 7, (h, w, 1)) + \
        np.sin(xx[..., None] * rng.uniform(0.1, 0.4) + yy[..., None] * 0.05) * 5
    out = out + tex
    out[lead] = (25, 22, 20)

    # border band + stonework outside the arch
    inner = lancet_mask(w, h)
    border_mask = lancet_mask(w, h) & ~lancet_mask_shrunk(w, h, int(size * 0.06))
    bcol = PALETTES[rng.choice(["red", "blue", "gold", "white"])][1]
    out[border_mask & ~lead] = jitter(bcol, 10)
    stone = np.array([95, 88, 78]) + np.random.normal(0, 6, (h, w, 1))
    out = np.where(inner[..., None], out, stone)

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


def lancet_mask_shrunk(w, h, px):
    m = lancet_mask(w, h)
    # erode by px using a cheap separable min filter
    from scipy.ndimage import binary_erosion
    return binary_erosion(m, iterations=px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.n):
        make_window(a.size, seed=a.seed * 1_000_000 + i).save(
            os.path.join(a.out, f"window_{i:05d}.png"))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{a.n}")


if __name__ == "__main__":
    main()
