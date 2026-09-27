#!/usr/bin/env python3
"""
Local web app: drop in a window photo, pick a pane, watch it shatter and the
network put it back together.

    python serve.py                      # then open http://localhost:8000
    python serve.py --ckpt checkpoints/shards.pt --port 8080

Uses only the standard library on top of the solver's own requirements.
The page is web/index.html; it talks to two endpoints here:
  POST /api/panes  (image bytes)  -> boxes of the single panes found in it
  POST /api/solve  (image bytes)  -> shards, the network's placements, score
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening

import shard_solver as ss
from shards import break_image, lead_map2, render_shards, snap_to_lead, snap_to_lead2
from split_panes import build_parser as pane_options, split_image

ROOT = os.path.dirname(os.path.abspath(__file__))
SCALE = 3          # shards are drawn at SCALE x the size the network works at
BLACK = 24         # grey level below which a pixel is see-through in the displayed shards
LEAD_MAX = 90      # ...and on a detected lead came, anything darker than this
LEAD_RGB = (55, 55, 55)   # colour of the lead drawn round each piece of glass
RIM = 3                   # its width in display pixels
_DISK = np.hypot(*np.mgrid[-RIM:RIM + 1, -RIM:RIM + 1]) <= RIM
_SPECK = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2   # see-through spots smaller than this stay glass
MAX_UPLOAD = 25 * 1024 * 1024


def png_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def edges(labels: np.ndarray) -> np.ndarray:
    e = np.zeros(labels.shape, bool)
    e[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    e[1:] |= labels[1:] != labels[:-1]
    return e


def overlay(mask: np.ndarray, rgb) -> np.ndarray:
    """Transparent RGBA image with `mask` pixels painted `rgb`."""
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask] = (*rgb, 255)
    return out


def smooth_upscale(labels: np.ndarray) -> np.ndarray:
    """SCALE x label map with smooth crack lines instead of pixel staircases."""
    h, w = labels.shape
    soft = [np.asarray(Image.fromarray((labels == i).astype(np.float32))
                       .resize((w * SCALE, h * SCALE), Image.BILINEAR))
            for i in range(labels.max() + 1)]
    return np.argmax(np.stack(soft), 0)


def display_shards(big: np.ndarray, labels: np.ndarray, lab: np.ndarray, angles, crop: int,
                   lead: np.ndarray | None = None):
    """Cut the same shards as render_shards, but from the SCALE x image (and the
    smooth SCALE x label map `lab`), so the page can draw them sharp. Same
    centroids and rotations as the network saw."""
    pad = crop // 2
    img_p = np.pad(big, ((pad, pad), (pad, pad), (0, 0)))
    lab_p = np.pad(lab, pad, constant_values=-1)
    if lead is not None:
        lead_p = np.pad(lead, pad)
    out = []
    for i, ang in enumerate(angles):
        ys, xs = np.nonzero(labels == i)
        y0 = int(round(ys.mean() * SCALE + (SCALE - 1) / 2))
        x0 = int(round(xs.mean() * SCALE + (SCALE - 1) / 2))
        rgb = img_p[y0:y0 + crop, x0:x0 + crop]
        m = (lab_p[y0:y0 + crop, x0:x0 + crop] == i).astype(np.float32)
        # for display only: near-black (lead lines, dark edges) fades to transparent,
        # so each shard reads as a loose piece of glass
        lum = rgb.astype(np.float32).mean(2)
        clear = lum < BLACK                      # truly black
        if lead is not None:                     # or a dark pixel on a detected lead came
            clear |= (lead_p[y0:y0 + crop, x0:x0 + crop] > 0.45) & (lum < LEAD_MAX)
        clear = binary_opening(clear, structure=_SPECK)   # ignore specks, keep real lines
        m *= ~clear                              # all or nothing: glass stays fully solid
        # lead around every remaining piece of glass (where the black was removed)
        # and along the shard's outline
        shape = lab_p[y0:y0 + crop, x0:x0 + crop] == i
        glass = m > 0                            # what is left after the black is removed
        rim = (shape & ~glass & binary_dilation(glass, structure=_DISK)) \
            | (shape & ~binary_erosion(shape, structure=_DISK, border_value=0))
        rgb = rgb.copy()
        rgb[rim] = LEAD_RGB
        m[rim] = 1
        im = Image.fromarray(np.dstack([rgb, (m * 255).astype(np.uint8)]), "RGBA")
        out.append(np.array(im.rotate(np.degrees(ang), resample=Image.BICUBIC)))
    return out


# one-line notes shown in the version selector, by version number
NOTES = {
    1: "synthetic windows only",
    2: "first real panes, trained on a laptop CPU",
    3: "GPU, 40 epochs",
    4: "GPU, 300 epochs",
    5: "overnight, scraped windows added",
    6: "after a cool-down",
    7: "trained on lead-only breaks",
    8: "20–28 shards",
}


def find_models(default_ckpt):
    """checkpoints/model_v<N>_*.pt, sorted by N, plus the --ckpt file if it isn't one."""
    import glob
    import re
    found = {}
    for path in glob.glob(os.path.join(ROOT, "checkpoints", "model_v*.pt")):
        m = re.match(r"model_v(\d+)", os.path.basename(path))
        if m:
            found.setdefault(int(m.group(1)), path)
    models = [dict(id=f"v{n}", label=f"v{n}", note=NOTES.get(n, os.path.basename(p)), path=p)
              for n, p in sorted(found.items())]
    default = next((m["id"] for m in models
                    if os.path.abspath(m["path"]) == os.path.abspath(default_ckpt)), None)
    if default is None and os.path.exists(default_ckpt):
        models.append(dict(id="custom", label=os.path.splitext(os.path.basename(default_ckpt))[0],
                           note="--ckpt", path=default_ckpt))
        default = "custom"
    return models, default or (models[-1]["id"] if models else None)


class Solver:
    def __init__(self, a):
        self.dev = ss.device(a)
        self.models, self.default = find_models(a.ckpt)
        if not self.models:
            raise SystemExit(f"no model found: put checkpoints/model_v*.pt in place or pass --ckpt")
        self.loaded = {}
        self.pane_opts = pane_options().parse_args([])

    def model(self, mid=None):
        mid = mid if any(m["id"] == mid for m in self.models) else self.default
        if mid not in self.loaded:
            path = next(m["path"] for m in self.models if m["id"] == mid)
            net = ss.ShardSolver().to(self.dev)
            net.load_state_dict(torch.load(path, weights_only=True, map_location=self.dev))
            self.loaded[mid] = net.eval()
        return mid, self.loaded[mid]

    def panes(self, data: bytes):
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        h, w = img.shape[:2]
        boxes = split_image(img, self.pane_opts)
        return dict(width=w, height=h,
                    panes=[dict(x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0))
                           for y0, y1, x0, x1 in boxes])

    @torch.no_grad()
    def solve(self, data: bytes, pieces: int, seed: int | None, lead: int, model_id=None):
        size, crop = ss.SIZE, ss.CROP
        mid, model = self.model(model_id)
        if seed is None:
            seed = int(np.random.default_rng().integers(2 ** 31))
        rng = np.random.default_rng(seed)
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img = np.array(pil.resize((size, size), Image.BICUBIC))
        big = np.array(pil.resize((size * SCALE, size * SCALE), Image.BICUBIC))
        labels = break_image(size, size, pieces, rng)
        if lead == 1:
            labels = snap_to_lead(img, labels, rng=rng)
        elif lead == 2:
            labels = snap_to_lead2(img, labels, rng=rng)
        s = render_shards(img, labels, crop, ss.TILE, rng=rng)
        K = len(s["tiles"])
        xy, cs = model(torch.from_numpy(s["tiles"])[None].to(self.dev),
                            torch.zeros(1, K, dtype=torch.bool, device=self.dev))
        xy, cs = xy[0].cpu().numpy(), cs[0].cpu().numpy()
        slot = ss.assign(xy, s["centers"])
        pred = np.arctan2(cs[:, 1], cs[:, 0])
        rot_err = ss.angle_err_deg(cs, s["angles"])
        big_lab = smooth_upscale(labels)
        lead_big = np.asarray(Image.fromarray(lead_map2(img).astype(np.float32))
                              .resize((size * SCALE, size * SCALE), Image.BILINEAR))
        shards = display_shards(big, labels, big_lab, s["angles"], crop * SCALE, lead_big)

        wrong = np.zeros(big_lab.shape, bool)
        for i in range(K):
            if slot[i] != i:
                wrong |= edges(big_lab == slot[i]) & (big_lab == slot[i])
        wrong = np.maximum.reduce([np.roll(wrong, d, ax) for ax in (0, 1) for d in (-1, 0, 1)])
        correct = int((slot == np.arange(K)).sum())
        D = size * SCALE
        return dict(
            size=D,
            image=png_url(big),
            cracks=png_url(overlay(edges(big_lab), (255, 255, 255))),
            wrong=png_url(overlay(wrong, (255, 255, 255))),
            shards=[dict(png=png_url(shards[i]),
                         home=[float(s["centers"][i][0] * D), float(s["centers"][i][1] * D)],
                         hole=[float(s["centers"][slot[i]][0] * D),
                               float(s["centers"][slot[i]][1] * D)],
                         angle=float(s["angles"][i]), pred_angle=float(pred[i]),
                         rot_err=float(rot_err[i]), correct=bool(slot[i] == i))
                    for i in range(K)],
            correct=correct, total=K, rot_median=float(np.median(rot_err)), seed=seed, model=mid,
            rot_within_15=float((rot_err < 15).mean()))


def make_handler(solver: Solver):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if urlparse(self.path).path == "/api/models":
                return self._json(dict(default=solver.default,
                                       models=[{k: m[k] for k in ("id", "label", "note")}
                                               for m in solver.models]))
            if urlparse(self.path).path in ("/", "/index.html"):
                with open(os.path.join(ROOT, "web", "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_UPLOAD:
                return self._json({"error": "send an image under 25 MB"}, 400)
            data = self.rfile.read(n)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/panes":
                    return self._json(solver.panes(data))
                if url.path == "/api/solve":
                    pieces = min(max(int(q.get("pieces", 12)), 4), 48)
                    seed = int(q["seed"]) if q.get("seed") else None
                    lead = {"1": 1, "2": 2}.get(q.get("lead"), 0)   # 0 random, 1 original, 2 improved
                    return self._json(solver.solve(data, pieces, seed, lead, q.get("model")))
                self._json({"error": "unknown endpoint"}, 404)
            except Exception as e:  # bad image etc. - report it on the page
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        def log_message(self, fmt, *args):
            print("  " + fmt % args)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model_v7_leadonly_10-16shards.pt")
    ap.add_argument("--size", type=int, default=192, help="must match the checkpoint")
    ap.add_argument("--crop", type=int, default=104, help="must match the checkpoint")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    ss.SIZE, ss.CROP = a.size, a.crop
    solver = Solver(a)
    names = ", ".join(m["label"] for m in solver.models)
    print(f"models: {names} (default {solver.default}) on {solver.dev} — open http://localhost:{a.port}")
    HTTPServer((a.host, a.port), make_handler(solver)).serve_forever()


if __name__ == "__main__":
    main()
