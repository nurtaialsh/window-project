#!/usr/bin/env python3
"""
Download stained-glass window photos from Wikimedia Commons for training.

Uses the official MediaWiki API (no page scraping). Walks the categories you
give it (and their sub-categories, to --depth), downloads each image at a
chosen width, skips small/badly shaped ones, and writes credits.csv with the
author + licence of every file so you can attribute them.

Run on your own computer:
    pip install requests pillow
    python download_windows.py --out data/real --max 400

Add your own categories with --cat "Category:Stained-glass windows in Hertfordshire"
(browse https://commons.wikimedia.org/wiki/Category:Stained-glass_windows to find more).

Tip: keep 20-30 windows aside in a separate folder (e.g. data/heldout) that are
NEVER used for training. Those are the ones to show on the call.
"""
import argparse
import csv
import io
import os
import re
import time

import requests
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "WindowShardSolver/0.1 (student research project; contact via GitHub)"}

DEFAULT_CATS = [  # names checked against Commons (note the hyphen in "Stained-glass")
    "Category:Featured pictures of stained-glass windows of churches",
    "Category:Medieval stained-glass windows in England",
    "Category:Stained-glass windows of Cathédrale Notre-Dame de Chartres",
    "Category:Stained-glass windows of the Sainte-Chapelle (Paris)",
    "Category:Stained-glass windows of Cologne Cathedral",
    "Category:Stained-glass windows of Seville Cathedral",
    "Category:Stained-glass windows of Augsburg Cathedral",
    "Category:Stained glass in the Victoria and Albert Museum",
]


def api(params):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(5):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited")


def members(cat, kind):
    cont = {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": cat,
                 "cmtype": kind, "cmlimit": 500, **cont})
        yield from (m["title"] for m in d["query"]["categorymembers"])
        if "continue" not in d:
            return
        cont = d["continue"]


def walk(cat, depth, seen):
    if cat in seen:
        return
    seen.add(cat)
    yield from members(cat, "file")
    if depth > 0:
        for sub in members(cat, "subcat"):
            yield from walk(sub, depth - 1, seen)


def info(titles, width):
    d = api({"action": "query", "prop": "imageinfo", "titles": "|".join(titles),
             "iiprop": "url|size|mime|extmetadata", "iiurlwidth": width})
    for p in d["query"]["pages"]:
        if "imageinfo" in p:
            yield p["title"], p["imageinfo"][0]


def clean(html):
    return re.sub("<[^>]+>", "", html or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--cat", action="append", help="category to add (repeatable)")
    ap.add_argument("--depth", type=int, default=1, help="how many sub-category levels")
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--width", type=int, default=1024, help="download width in px")
    ap.add_argument("--min-side", type=int, default=800, help="skip originals smaller than this")
    ap.add_argument("--max-aspect", type=float, default=3.5, help="skip very long/thin images")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cats = a.cat or DEFAULT_CATS
    credits_path = os.path.join(a.out, "credits.csv")
    new = not os.path.exists(credits_path)
    cf = open(credits_path, "a", newline="", encoding="utf-8")
    w = csv.writer(cf)
    if new:
        w.writerow(["file", "title", "author", "licence", "source"])

    seen_cats, got = set(), 0
    titles = []
    for c in cats:
        try:
            titles += [t for t in walk(c, a.depth, seen_cats) if t not in titles]
        except Exception as e:
            print("skip", c, e)
        print(f"{c}: {len(titles)} candidates so far")

    for i in range(0, len(titles), 40):
        if got >= a.max:
            break
        for title, ii in info(titles[i:i + 40], a.width):
            if got >= a.max:
                break
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            W, H = ii["width"], ii["height"]
            if min(W, H) < a.min_side or max(W, H) / min(W, H) > a.max_aspect:
                continue
            name = re.sub(r"[^\w.-]+", "_", title.split(":", 1)[1])[:120]
            name = os.path.splitext(name)[0] + ".jpg"
            path = os.path.join(a.out, name)
            if os.path.exists(path):
                continue
            try:
                r = requests.get(ii.get("thumburl") or ii["url"], headers=HEADERS, timeout=60)
                r.raise_for_status()
                Image.open(io.BytesIO(r.content)).convert("RGB").save(path, quality=92)
            except Exception as e:
                print("fail", title, e)
                continue
            md = ii.get("extmetadata", {})
            w.writerow([name, title, clean(md.get("Artist", {}).get("value")),
                        clean(md.get("LicenseShortName", {}).get("value")),
                        ii.get("descriptionurl")])
            got += 1
            print(f"[{got}] {name}")
            time.sleep(0.3)  # be polite
    cf.close()
    print(f"done: {got} images in {a.out}. Now open the folder and delete any that are "
          "not a clear, straight-on view of a window.")


if __name__ == "__main__":
    main()
