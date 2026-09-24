#!/usr/bin/env python3
"""
Cut photos of multi-light windows into single panes by removing the black.

Stone mullions, transoms and the wall around a window photograph as nearly
black bands that run right across the image. We find rows/columns that are
mostly dark, cut along them, and recurse into each piece (columns, then rows,
then columns...), so a four-light window with tracery comes apart into its
four lights plus the tracery openings. Small or mostly-dark pieces are
dropped, and tall lancets are cut into near-square tiles so they don't get
squashed when the solver resizes them to a square.

Usage
  python split_panes.py --images data/real --out data/panes
  python split_panes.py --images data/real --out data/panes --preview previews/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter1d

from shard_solver import list_images

WORK = 480  # analyse at this width; crops are taken from the full-res image


def dark_mask(img: np.ndarray, thresh: float) -> np.ndarray:
    g = img.astype(np.float32).mean(2)
    return g < thresh


def _runs(keep: np.ndarray):
    """(start, end) of each run of True."""
    d = np.diff(np.r_[0, keep.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def segments(prof, a):
    """Pieces left after cutting along dark bands at least min_gap wide.
    Thinner dark lines are lead and saddle bars inside a pane, not stonework."""
    keep = np.ones(len(prof), bool)
    for s, e in _runs(prof > a.cut_frac):
        if e - s >= a.min_gap:
            keep[s:e] = False
    return [(s, e) for s, e in _runs(keep) if e - s >= a.min_px]


def split_region(dark, box, axis, depth, a, out):
    """Recursively cut box=(y0, y1, x0, x1) along mostly-dark lines."""
    y0, y1, x0, x1 = box
    sub = dark[y0:y1, x0:x1]
    h, w = sub.shape
    if h < a.min_px or w < a.min_px:
        return
    # trim dark margins on all sides first
    rows = uniform_filter1d(sub.mean(1), 3) > a.edge_frac
    cols = uniform_filter1d(sub.mean(0), 3) > a.edge_frac
    if rows.all() or cols.all():
        return
    ry, cx = np.nonzero(~rows)[0], np.nonzero(~cols)[0]
    y0, y1, x0, x1 = y0 + ry[0], y0 + ry[-1] + 1, x0 + cx[0], x0 + cx[-1] + 1
    sub = dark[y0:y1, x0:x1]
    # look for dark bands across the whole region along the current axis
    prof = uniform_filter1d(sub.mean(0 if axis == 1 else 1), 3)
    parts = segments(prof, a)
    if len(parts) > 1 and depth > 0:
        for s, e in parts:
            nb = (y0, y1, x0 + s, x0 + e) if axis == 1 else (y0 + s, y0 + e, x0, x1)
            split_region(dark, nb, 1 - axis, depth - 1, a, out)
        return
    if depth > 0:  # nothing to cut this way; try the other axis once
        prof2 = uniform_filter1d(sub.mean(1 if axis == 1 else 0), 3)
        parts2 = segments(prof2, a)
        if len(parts2) > 1:
            for s, e in parts2:
                nb = (y0 + s, y0 + e, x0, x1) if axis == 1 else (y0, y1, x0 + s, x0 + e)
                split_region(dark, nb, axis, depth - 1, a, out)
            return
    if sub.mean() <= a.max_dark:
        out.append((y0, y1, x0, x1))


def tile(box, max_aspect):
    """Cut a long box into near-square, evenly spaced (possibly overlapping) tiles."""
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    if max(h, w) <= max_aspect * min(h, w):
        return [box]
    side, length = min(h, w), max(h, w)
    n = int(np.ceil(length / side))
    starts = np.linspace(0, length - side, n).round().astype(int)
    if h > w:
        return [(y0 + s, y0 + s + side, x0, x1) for s in starts]
    return [(y0, y1, x0 + s, x0 + s + side) for s in starts]


def split_image(img: np.ndarray, a):
    H, W = img.shape[:2]
    scale = WORK / W
    small = np.array(Image.fromarray(img).resize((WORK, max(1, round(H * scale))), Image.BILINEAR))
    dark = dark_mask(small, a.thresh)
    boxes = []
    split_region(dark, (0, dark.shape[0], 0, dark.shape[1]), 1, a.depth, a, boxes)
    out = []
    for b in boxes:
        for y0, y1, x0, x1 in tile(b, a.max_aspect):
            if dark[y0:y1, x0:x1].mean() > a.max_dark:
                continue
            fy = lambda v: int(round(v / scale))
            out.append((fy(y0), min(H, fy(y1)), fy(x0), min(W, fy(x1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", help="folder for images with the cut boxes drawn on")
    ap.add_argument("--thresh", type=float, default=40, help="grey level counted as black")
    ap.add_argument("--cut-frac", type=float, default=0.8,
                    help="a line this dark (fraction of black pixels) is a mullion/transom")
    ap.add_argument("--edge-frac", type=float, default=0.6, help="trim margins darker than this")
    ap.add_argument("--max-dark", type=float, default=0.3, help="drop panes darker than this")
    ap.add_argument("--min-px", type=int, default=48, help="smallest pane side, at 480 px width")
    ap.add_argument("--min-gap", type=int, default=6,
                    help="thinnest dark band (px at 480 width) treated as stonework")
    ap.add_argument("--max-aspect", type=float, default=1.5, help="tile panes longer than this")
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.preview:
        os.makedirs(a.preview, exist_ok=True)
    total = 0
    for p in list_images(a.images):
        img = np.array(Image.open(p).convert("RGB"))
        boxes = split_image(img, a)
        stem = os.path.splitext(os.path.basename(p))[0]
        for k, (y0, y1, x0, x1) in enumerate(boxes):
            Image.fromarray(img[y0:y1, x0:x1]).save(
                os.path.join(a.out, f"{stem}__{k:02d}.jpg"), quality=92)
        total += len(boxes)
        if a.preview:
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            for y0, y1, x0, x1 in boxes:
                d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=4)
            im.thumbnail((800, 800))
            im.save(os.path.join(a.preview, stem + ".jpg"), quality=85)
        print(f"{len(boxes):3d} panes  {os.path.basename(p)}")
    print(f"done: {total} panes in {a.out}")


if __name__ == "__main__":
    main()
