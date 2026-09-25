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

## Web app
    python serve.py            # then open http://localhost:8000

Drop in a window photo; multi-light windows are split into panes you can click. The
network shatters the pane, and the page animates each shard flying to the hole the
network chose, turned by the angle it predicted, with the score underneath. It uses
`checkpoints/model_v2_realpanes_10-16shards.pt` by default (`--ckpt` for another; pass
`--size/--crop` if that one was trained with different settings).

## Training on real windows (NVIDIA GPU)
    pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build of torch
    pip install numpy scipy pillow scikit-image requests
    python -c "import torch; print(torch.cuda.is_available())"             # must print True
    python split_panes.py --images data/real --out data/panes
    copy checkpoints\model_v2_realpanes_10-16shards.pt checkpoints\shards.pt   # start from v2
    python shard_solver.py train --images data/panes --resume --epochs 40 --lr 5e-4 --workers 4 --size 192 --kmin 10 --kmax 16 --crop 104

`split_panes.py` cuts multi-light windows into single panes along the dark stonework.
Training uses the GPU automatically when there is one (`--device cpu` to force CPU).
Pass the same `--size/--kmin/--kmax/--crop` to `demo` and `eval` as you trained with.

## Status
- `checkpoints/model_v1_synthetic_10-16shards.pt`: trained on synthetic windows
  (`make_windows.py`), 10–16 shards, 192 px. On 55 held-out synthetic windows:
  ~87% of shards placed correctly, ~45% of windows perfect, median rotation error ~7°.
  Lead-following cracks score about the same (86% / 44% / 6°) without retraining.
- `checkpoints/model_v2_realpanes_10-16shards.pt`: v1 fine-tuned for 30 epochs (~9 h on
  4 CPU cores) on 3,247 single panes cut by `split_panes.py` from 322 real window photos
  (real.zip, 46 unusable photos removed). On panes from 20 windows never seen in training
  (192 px, 10–16 shards): ~42% of shards placed correctly (v1: ~8%, chance ~8%),
  median rotation error ~20° (v1: ~33°). Resume from this one when training further.
- Next: more epochs on a GPU, more real windows (`download_windows.py`), then 288 px with
  24–48 shards.
- `legacy/` — first version, which cut windows on a square grid.
