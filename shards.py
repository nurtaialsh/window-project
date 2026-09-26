"""
Break a window image into irregular, crack-like shards and scatter them.

    labels = break_image(h, w, k)       # HxW int map, one id per shard
    shards = render_shards(img, labels) # per-shard rotated crops + targets

Crack lines come from a Voronoi diagram computed on a warped coordinate
grid: the smooth warp bends the straight Voronoi edges into wandering
fracture lines, and a small high-frequency warp makes them jagged.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import zoom, label as cc_label
from scipy.spatial import cKDTree


def _smooth_noise(h, w, cells, amp, rng):
    g = rng.normal(0, 1, (cells + 1, cells + 1))
    z = zoom(g, ((h + cells) / (cells + 1), (w + cells) / (cells + 1)), order=3)
    return z[:h, :w] * amp


def _seeds(h, w, k, rng):
    """Roughly even random points (dart throwing), so shards are similar-ish in size."""
    min_d = 0.6 * np.sqrt(h * w / k)
    pts = []
    tries = 0
    while len(pts) < k:
        p = rng.uniform([0, 0], [w, h])
        if all(np.hypot(*(p - q)) >= min_d for q in pts) or tries > 5000:
            pts.append(p)
        tries += 1
    return np.array(pts)


def break_image(h, w, k, rng=None):
    rng = rng or np.random.default_rng()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = min(h, w)
    dx = _smooth_noise(h, w, 4, 0.06 * s, rng) + _smooth_noise(h, w, 24, 0.012 * s, rng)
    dy = _smooth_noise(h, w, 4, 0.06 * s, rng) + _smooth_noise(h, w, 24, 0.012 * s, rng)
    pts = _seeds(h, w, k, rng)
    _, idx = cKDTree(pts).query(np.column_stack([(xx + dx).ravel(), (yy + dy).ravel()]))
    labels = idx.reshape(h, w)
    # the warp can occasionally strand a tiny island; merge those into a neighbour
    for i in range(k):
        comp, n = cc_label(labels == i)
        if n > 1:
            sizes = np.bincount(comp.ravel())[1:]
            keep = np.argmax(sizes) + 1
            for c in range(1, n + 1):
                if c != keep:
                    m = comp == c
                    ys, xs = np.nonzero(m)
                    y, x = ys[0], xs[0]
                    nb = labels[max(0, y - 1), x] if labels[max(0, y - 1), x] != i else \
                        labels[min(h - 1, y + 1), x]
                    labels[m] = nb
    # relabel to 0..K-1 (a seed may have lost everything)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(h, w)


def lead_map(img: np.ndarray) -> np.ndarray:
    """0..1 map that is high on dark, thin lines (lead cames, painted outlines)."""
    from scipy.ndimage import gaussian_filter, grey_closing
    g = img.astype(np.float32).mean(2) / 255.0
    # "black-hat": how much darker a pixel is than its surroundings -> thin dark lines
    s = max(3, int(min(g.shape) * 0.012)) | 1
    tophat = grey_closing(g, size=(s, s)) - g
    dark = np.clip((0.35 - g) / 0.35, 0, 1)      # absolutely dark pixels too
    m = np.maximum(tophat / (tophat.max() + 1e-6), dark)
    return gaussian_filter(m, 0.8)


def lead_map2(img: np.ndarray, min_len_frac=0.06) -> np.ndarray:
    """0..1 map of lead cames, stricter than lead_map: a ridge filter keeps only dark
    *line-shaped* structures of came width (so dark glass isn't counted), and only
    long connected networks survive (so short paint strokes and texture drop out)."""
    from scipy.ndimage import gaussian_filter
    from skimage.filters import sato
    from skimage.morphology import skeletonize
    g = img.astype(np.float32).mean(2) / 255.0
    s = min(g.shape)
    r = sato(g, sigmas=[max(1.0, s * f) for f in (0.004, 0.007, 0.011)], black_ridges=True)
    r = r / (np.percentile(r, 99.5) + 1e-6)
    m = r > 0.18
    lab, n = cc_label(m, structure=np.ones((3, 3)))
    if n:
        length = np.bincount(lab[skeletonize(m)], minlength=n + 1)
        keep = length >= min_len_frac * s
        keep[0] = False
        m = keep[lab]
    out = gaussian_filter(np.clip(r, 0, 1) * m, 1.0)
    return out / (out.max() + 1e-6)


def snap_to_lead(img: np.ndarray, labels: np.ndarray, band_frac=0.045,
                 rng=None, lead_fn=None, noise=0.05) -> np.ndarray:
    """Move crack lines onto nearby lead lines.

    The random cracks from break_image() are kept as a rough plan. Each shard's
    core (everything further than `band` pixels from a crack) is fixed; inside
    the band around each crack a watershed lets the boundary slide to the
    darkest nearby line. Where there is no lead line close by, the boundary
    stays roughly where it was and breaks straight through the glass - like a
    real fracture.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed
    rng = rng or np.random.default_rng()
    h, w = labels.shape
    band = max(2, int(min(h, w) * band_frac))
    edge = np.zeros_like(labels, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    dist = distance_transform_edt(~edge)
    from scipy.ndimage import gaussian_filter, maximum_position
    markers = np.where(dist > band, labels + 1, 0)
    # small shards may have no core that far from a crack: seed them at their
    # innermost point instead, so no shard disappears
    k = labels.max() + 1
    have = np.zeros(k + 1, bool)
    have[np.unique(markers)] = True
    for i in np.nonzero(~have[1:])[0]:
        y, x = maximum_position(dist, labels, i)
        r = max(1, int(dist[y, x] * 0.6))
        sl = (slice(max(0, y - r), y + r + 1), slice(max(0, x - r), x + r + 1))
        markers[sl][labels[sl] == i] = i + 1
    # elevation: high on lead; smooth noise so cracks through plain glass wander
    wobble = gaussian_filter(rng.normal(0, 1, labels.shape), 1.5)
    elev = (lead_fn or lead_map)(img) + noise * wobble.astype(np.float32) / (wobble.std() + 1e-6)
    # a gentle pull back toward the original crack keeps shard sizes balanced
    elev += 0.15 * (1 - np.clip(dist / band, 0, 1))
    out = watershed(elev, markers)
    return out - 1


def render_shards(img: np.ndarray, labels: np.ndarray, crop: int, out: int,
                  rotate=True, rng=None):
    """Cut each shard out, rotate it about its centroid by a random angle.

    Returns dict with
      tiles   (K, 4, out, out) float32 : RGB * mask, mask   (what the model sees)
      centers (K, 2)  float32 : true centroid (x, y), normalised to [0,1]
      angles  (K,)    float32 : rotation applied, radians
      rgba    list of (crop, crop, 4) uint8 full-res rotated shards (for drawing)
    """
    rng = rng or np.random.default_rng()
    h, w = labels.shape
    k = labels.max() + 1
    pad = crop // 2
    img_p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    lab_p = np.pad(labels, pad, constant_values=-1)
    tiles, centers, angles, rgbas = [], [], [], []
    for i in range(k):
        ys, xs = np.nonzero(labels == i)
        cy, cx = ys.mean(), xs.mean()
        y0, x0 = int(round(cy)), int(round(cx))
        rgb = img_p[y0:y0 + crop, x0:x0 + crop]
        m = (lab_p[y0:y0 + crop, x0:x0 + crop] == i).astype(np.uint8) * 255
        rgba = np.dstack([rgb, m])
        ang = rng.uniform(0, 2 * np.pi) if rotate else 0.0
        im = Image.fromarray(rgba, "RGBA").rotate(np.degrees(ang), resample=Image.BILINEAR)
        rgba = np.array(im)
        small = np.array(im.resize((out, out), Image.BILINEAR)).astype(np.float32) / 255
        a = small[..., 3:4]
        tiles.append(np.concatenate([small[..., :3] * a, a], axis=2).transpose(2, 0, 1))
        centers.append([cx / w, cy / h])
        angles.append(ang)
        rgbas.append(rgba)
    return dict(tiles=np.stack(tiles).astype(np.float32),
                centers=np.array(centers, np.float32),
                angles=np.array(angles, np.float32),
                rgba=rgbas)


def snap_to_lead2(img: np.ndarray, labels: np.ndarray, rng=None, leadmap=None) -> np.ndarray:
    """snap_to_lead with the stricter lead_map2, a wider reach and less wobble, so
    cracks follow the lead cames cleanly and run straighter through plain glass.
    Pass a precomputed lead_map2(img) as leadmap to skip the (slow) detection."""
    fn = lead_map2 if leadmap is None else (lambda _: leadmap)
    return snap_to_lead(img, labels, band_frac=0.07, rng=rng, lead_fn=fn, noise=0.015)
