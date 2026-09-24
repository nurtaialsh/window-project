# Window shard solver

Shatters an image of a stained-glass window into irregular shards, scatters and
rotates them, and a neural network works out where each shard belongs and how
far to turn it back — from what is painted on it.

Motivation: Winchester Cathedral's Great West Window was smashed in the 1640s and
its fragments later reset as a mosaic. This is a first step toward using machine
learning to suggest where such fragments originally belonged.

## How it works
1. **Breaking** (`shards.py`) — random crack lines (a Voronoi diagram on a warped
   grid). `snap_to_lead` moves cracks onto nearby lead lines, so pieces come apart
   along the lead where it's close and break through the glass where it isn't.
2. **The network** (`shard_solver.py`) — a small CNN encodes each shard; a
   transformer lets the shards compare with each other; it predicts each shard's
   original centre (x, y) and rotation (as a direction on a circle).
3. **Solving** — the Hungarian algorithm gives each shard its own hole, then each
   shard is turned back by its predicted angle.

## Usage
    pip install -r requirements.txt
    python download_windows.py --out data/real --max 2000 --width 800 --depth 2
    python shard_solver.py train --images data/real --minutes 120
    python shard_solver.py demo  --image path/to/window.jpg --out result.png
    python shard_solver.py eval  --images data/heldout

## Status
- `checkpoints/model_v1_synthetic_10-16shards.pt`: trained on synthetic windows
  (`make_windows.py`), 10–16 shards, 192 px. On 55 held-out synthetic windows:
  ~87% of shards placed correctly, ~45% of windows perfect, median rotation error ~7°.
  Lead-following cracks score about the same (86% / 44% / 6°) without retraining.
- Next: train on real cathedral photos at 512 px with 24–48 shards.
- `legacy/` — first version, which cut windows on a square grid.
