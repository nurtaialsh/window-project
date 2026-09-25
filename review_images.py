#!/usr/bin/env python3
"""
Sort out scraped window photos quickly: a local page shows every photo as a
thumbnail, most suspicious first, with the clearly bad ones already marked.
Click a photo to flip its mark, then move the marked ones out of the folder.

    python review_images.py --images data\\more        # then open http://localhost:8001

Marked automatically:
  * likely not a clear view of a window - a small classifier on colour,
    contrast and sharpness, fitted to 388 hand-sorted photos from real.zip
    (when it is confident it is usually right, but it misses about half of
    the bad photos, so skim the top of the list yourself)
  * black-and-white photos
  * near-duplicates of a photo earlier in the list
Rejected photos are moved to --rejected (default: data/rejected), not deleted.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
from scipy.ndimage import laplace, uniform_filter

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")
# logistic regression fitted on real.zip: 46 photos removed by hand, 342 kept
MU = np.array([24.84, 0.02466, 0.1468, 38.01, 0.4079, 8.768, 0.5064, 0.08847, 1.586])
SD = np.array([11.79, 0.06671, 0.09418, 16.19, 0.1701, 0.5906, 0.1248, 0.07679, 0.4682])
W = np.array([0.07739, -0.2188, 0.02748, -1.116, -0.6406, -0.39, -0.6346, 0.2998, -0.2299])
B = -0.7297
MARK_ABOVE = 0.8     # classifier confidence at which a photo is pre-marked
GREY_SAT = 8         # mean channel spread below this = black-and-white photo
DUP_BITS = 20        # difference-hash distance (of 256 bits) counted as a duplicate


def features(im: Image.Image) -> np.ndarray:
    w0, h0 = im.size
    s = np.asarray(im.resize((256, max(1, round(h0 * 256 / w0))))).astype(np.float32)
    g = s.mean(2)
    sat = s.max(2) - s.min(2)
    m = uniform_filter(g, 8)
    std = np.sqrt(np.clip(uniform_filter(g * g, 8) - m * m, 0, None))
    lap = laplace(g)
    rg = s[..., 0] - s[..., 1]
    yb = 0.5 * (s[..., 0] + s[..., 1]) - s[..., 2]
    return np.array([
        sat.mean(),                                           # colour
        ((std < 6) & (g > 40) & (g < 215)).mean(),            # plain wall / stone / sky
        ((sat < 20) & (g > 60) & (g < 200)).mean(),           # colourless mid-tones
        np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()),  # colourfulness
        (g < 30).mean(),                                      # black
        np.log1p(lap.var()),                                  # sharpness
        (np.abs(lap) > 20).mean(),                            # busy detail (lead, paint)
        ((sat > 60) & (g > 60)).mean(),                       # lit coloured glass
        max(w0, h0) / min(w0, h0),
    ])


def dhash(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L").resize((17, 16))).astype(int)
    return (g[:, 1:] > g[:, :-1]).ravel()


def analyse(folder: str, cache_path: str):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    files = sorted({os.path.basename(p) for e in EXTS for p in glob.glob(os.path.join(folder, e))})
    items = []
    hashes = np.zeros((len(files), 256), bool)   # rows of earlier readable photos
    owner = []                                   # item index for each filled row
    for n, name in enumerate(files, 1):
        path = os.path.join(folder, name)
        key = f"{name}|{os.path.getmtime(path)}"
        if key not in cache:
            try:
                im = Image.open(path).convert("RGB")
                x = features(im)
                cache[key] = dict(x=x.tolist(), h=dhash(im).astype(int).tolist())
            except Exception as e:
                cache[key] = dict(error=str(e))
            if n % 50 == 0:
                print(f"  looked at {n}/{len(files)}", flush=True)
        c = cache[key]
        if "error" in c:
            items.append(dict(f=name, score=1.0, reasons=["unreadable"], reject=True))
            continue
        x = np.array(c["x"])
        score = float(1 / (1 + np.exp(-(((x - MU) / SD) @ W + B))))
        reasons = []
        if score > MARK_ABOVE:
            reasons.append("probably not a clear window")
        if x[0] < GREY_SAT:
            reasons.append("black and white")
        h = np.array(c["h"], bool)
        if owner:
            d = (hashes[:len(owner)] != h).sum(1)
            j = int(d.argmin())
            if d[j] <= DUP_BITS:
                reasons.append(f"duplicate of {items[owner[j]]['f']}")
        hashes[len(owner)] = h
        owner.append(len(items))
        items.append(dict(f=name, score=score, reasons=reasons, reject=bool(reasons)))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    items.sort(key=lambda it: (-bool(it["reasons"]), -it["score"]))
    return items


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo Review</title><link rel="icon" href="data:,">
<style>
  :root { --bg:#111114; --panel:#1b1b20; --line:#2c2c34; --text:#ececf1; --muted:#9a9aa8; --bad:#ff5a5a; --accent:#e0b44c; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; z-index:2; background:rgba(17,17,20,.95); border-bottom:1px solid var(--line);
           padding:12px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  h1 { font-size:17px; margin:0 12px 0 0; }
  .muted { color:var(--muted); }
  button { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:7px; padding:7px 12px; cursor:pointer; font-size:14px; }
  button.on { border-color:var(--accent); color:var(--accent); }
  button.go { background:var(--bad); border-color:var(--bad); color:#fff; font-weight:600; margin-left:auto; }
  button:disabled { opacity:.4; cursor:default; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:10px; padding:14px 16px 40px; }
  .card { position:relative; background:var(--panel); border:2px solid var(--line); border-radius:8px; overflow:hidden; cursor:pointer; }
  .card img { width:100%; aspect-ratio:1; object-fit:contain; background:#000; display:block; }
  .card.rej { border-color:var(--bad); }
  .card.rej img { opacity:.35; }
  .card.rej::after { content:"✕ reject"; position:absolute; top:6px; left:6px; background:var(--bad); color:#fff;
                     font-weight:700; font-size:12px; padding:2px 7px; border-radius:5px; }
  .meta { padding:6px 8px; font-size:12px; }
  .meta .why { color:var(--bad); }
  .meta .name { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .bar { height:3px; background:linear-gradient(90deg, var(--bad) var(--p), transparent var(--p)); }
</style></head><body>
<header>
  <h1>Photo review</h1>
  <span class="muted" id="count"></span>
  <button id="f-all" class="on">All</button>
  <button id="f-rej">Marked</button>
  <button id="f-keep">Kept</button>
  <span class="muted">Click a photo to mark/unmark · Shift+click to open it full size · red bar = how suspicious it looks</span>
  <button id="move" class="go" disabled>Move marked out</button>
</header>
<div id="grid"></div>
<script>
let items = [], filter = "all";
const grid = document.getElementById("grid");
async function load() {
  grid.innerHTML = '<p class="muted">Analysing photos…</p>';
  items = await (await fetch("/api/list")).json();
  render();
}
function render() {
  grid.innerHTML = "";
  for (const it of items) {
    if (filter === "rej" && !it.reject) continue;
    if (filter === "keep" && it.reject) continue;
    const c = document.createElement("div");
    c.className = "card" + (it.reject ? " rej" : "");
    c.innerHTML = `<img loading="lazy" src="/img?f=${encodeURIComponent(it.f)}&s=320" alt="">
      <div class="bar" style="--p:${Math.round(100 * it.score)}%"></div>
      <div class="meta"><div class="why">${it.reasons.join(" · ")}</div><div class="name" title="${it.f}">${it.f}</div></div>`;
    c.onclick = e => {
      if (e.shiftKey) { window.open(`/img?f=${encodeURIComponent(it.f)}&s=0`); return; }
      it.reject = !it.reject; c.classList.toggle("rej", it.reject); count();
    };
    grid.appendChild(c);
  }
  count();
}
function count() {
  const n = items.filter(i => i.reject).length;
  document.getElementById("count").textContent = `${items.length} photos · ${n} marked`;
  const b = document.getElementById("move"); b.disabled = !n; b.textContent = `Move ${n} marked out`;
}
for (const [id, f] of [["f-all", "all"], ["f-rej", "rej"], ["f-keep", "keep"]])
  document.getElementById(id).onclick = e => {
    filter = f; document.querySelectorAll("header button[id^=f-]").forEach(b => b.classList.toggle("on", b === e.target)); render();
  };
document.getElementById("move").onclick = async () => {
  const files = items.filter(i => i.reject).map(i => i.f);
  if (!confirm(`Move ${files.length} photos to the rejected folder?`)) return;
  const r = await (await fetch("/api/move", { method: "POST", body: JSON.stringify({ files }) })).json();
  if (r.error) { alert(r.error); return; }
  items = items.filter(i => !i.reject); render();
  alert(`Moved ${r.moved} photos to ${r.to}`);
};
load();
</script></body></html>"""


def make_handler(a):
    thumbs = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _safe(self, name):
            name = os.path.basename(name or "")
            path = os.path.join(a.images, name)
            return path if name and os.path.isfile(path) else None

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if url.path == "/api/list":
                return self._json(analyse(a.images, os.path.join(a.images, ".review_cache.json")))
            if url.path == "/img":
                path = self._safe(q.get("f"))
                if not path:
                    return self._send(404, b"not found", "text/plain")
                size = int(q.get("s", 320))
                if size == 0:
                    ext = os.path.splitext(path)[1].lower().lstrip(".")
                    ctype = {"png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
                    with open(path, "rb") as f:
                        return self._send(200, f.read(), ctype)
                if path not in thumbs:
                    im = Image.open(path).convert("RGB")
                    im.thumbnail((size, size))
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=80)
                    thumbs[path] = buf.getvalue()
                return self._send(200, thumbs[path], "image/jpeg")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/move":
                return self._json({"error": "unknown endpoint"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            files = json.loads(self.rfile.read(n) or b"{}").get("files", [])
            os.makedirs(a.rejected, exist_ok=True)
            moved = 0
            for name in files:
                path = self._safe(name)
                if path:
                    shutil.move(path, os.path.join(a.rejected, os.path.basename(path)))
                    thumbs.pop(path, None)
                    moved += 1
            print(f"  moved {moved} photos to {a.rejected}")
            self._json({"moved": moved, "to": os.path.abspath(a.rejected)})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of photos to review")
    ap.add_argument("--rejected", default=os.path.join("data", "rejected"))
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    print("checking photos (first time takes a minute or two for a few hundred)...")
    items = analyse(a.images, os.path.join(a.images, ".review_cache.json"))
    print(f"{len(items)} photos, {sum(i['reject'] for i in items)} marked. "
          f"Open http://localhost:{a.port}")
    HTTPServer(("127.0.0.1", a.port), make_handler(a)).serve_forever()


if __name__ == "__main__":
    main()
