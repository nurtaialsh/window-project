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

import shard_solver as ss
from shards import break_image, render_shards, snap_to_lead, snap_to_lead2
from split_panes import build_parser as pane_options, split_image

ROOT = os.path.dirname(os.path.abspath(__file__))
SCALE = 3          # shards are drawn at SCALE x the size the network works at
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


def display_shards(big: np.ndarray, labels: np.ndarray, lab: np.ndarray, angles, crop: int):
    """Cut the same shards as render_shards, but from the SCALE x image (and the
    smooth SCALE x label map `lab`), so the page can draw them sharp. Same
    centroids and rotations as the network saw."""
    pad = crop // 2
    img_p = np.pad(big, ((pad, pad), (pad, pad), (0, 0)))
    lab_p = np.pad(lab, pad, constant_values=-1)
    out = []
    for i, ang in enumerate(angles):
        ys, xs = np.nonzero(labels == i)
        y0 = int(round(ys.mean() * SCALE + (SCALE - 1) / 2))
        x0 = int(round(xs.mean() * SCALE + (SCALE - 1) / 2))
        rgb = img_p[y0:y0 + crop, x0:x0 + crop]
        m = (lab_p[y0:y0 + crop, x0:x0 + crop] == i).astype(np.uint8) * 255
        im = Image.fromarray(np.dstack([rgb, m]), "RGBA")
        out.append(np.array(im.rotate(np.degrees(ang), resample=Image.BICUBIC)))
    return out


class Solver:
    def __init__(self, a):
        self.dev = ss.device(a)
        self.model = ss.ShardSolver().to(self.dev)
        self.model.load_state_dict(torch.load(a.ckpt, weights_only=True, map_location=self.dev))
        self.model.eval()
        self.pane_opts = pane_options().parse_args([])
        self.lead_method = a.lead_method

    def panes(self, data: bytes):
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        h, w = img.shape[:2]
        boxes = split_image(img, self.pane_opts)
        return dict(width=w, height=h,
                    panes=[dict(x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0))
                           for y0, y1, x0, x1 in boxes])

    @torch.no_grad()
    def solve(self, data: bytes, pieces: int, seed: int | None, lead: bool):
        size, crop = ss.SIZE, ss.CROP
        rng = np.random.default_rng(seed)
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img = np.array(pil.resize((size, size), Image.BICUBIC))
        big = np.array(pil.resize((size * SCALE, size * SCALE), Image.BICUBIC))
        labels = break_image(size, size, pieces, rng)
        if lead:
            snapper = snap_to_lead2 if self.lead_method == 2 else snap_to_lead
            labels = snapper(img, labels, rng=rng)
        s = render_shards(img, labels, crop, ss.TILE, rng=rng)
        K = len(s["tiles"])
        xy, cs = self.model(torch.from_numpy(s["tiles"])[None].to(self.dev),
                            torch.zeros(1, K, dtype=torch.bool, device=self.dev))
        xy, cs = xy[0].cpu().numpy(), cs[0].cpu().numpy()
        slot = ss.assign(xy, s["centers"])
        pred = np.arctan2(cs[:, 1], cs[:, 0])
        rot_err = ss.angle_err_deg(cs, s["angles"])
        big_lab = smooth_upscale(labels)
        shards = display_shards(big, labels, big_lab, s["angles"], crop * SCALE)

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
            correct=correct, total=K, rot_median=float(np.median(rot_err)),
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
                    return self._json(solver.solve(data, pieces, seed, q.get("lead") == "1"))
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
    ap.add_argument("--lead-method", type=int, default=2, choices=(1, 2),
                    help="how cracks follow the lead: 1 = original, 2 = stricter detection")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    ss.SIZE, ss.CROP = a.size, a.crop
    solver = Solver(a)
    print(f"model {a.ckpt} on {solver.dev} — open http://localhost:{a.port}")
    HTTPServer((a.host, a.port), make_handler(solver)).serve_forever()


if __name__ == "__main__":
    main()
