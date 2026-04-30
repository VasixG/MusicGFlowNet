# GFlowNet music loop with torchgfn

Tiny multi-track MIDI loop generator trained with `torchgfn`, not a hand-written GFlowNet loss.
It creates an 8-bar grid loop with drums, bass, chords, and lead, then exports a MIDI file and piano-roll image.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_torchgfn_music.py --steps 3000 --batch 64 --device cuda --out out
```

CPU smoke test:

```bash
python train_torchgfn_music.py --steps 200 --batch 16 --device cpu --out out_cpu
```

Outputs:

- `best_sample.mid` — listen in any DAW / VLC / GarageBand / Ableton / MuseScore
- `best_sample.png` — piano-roll visualization
- `loss.png` — training loss curve
- `model.pt` — trained weights and config

## Notes

This is intentionally small and discrete. Each action fills the next grid cell with one token.
The GFlowNet objective is `TBGFlowNet` from `torchgfn`.

## Notes on loss scale

Trajectory Balance loss contains a sum of log-probabilities over the whole trajectory. For an 8-bar loop this is roughly 512 token decisions, so initializing `logZ=0` makes the initial loss enormous. This version uses `--logz-init auto`, approximately `n_cells * log(n_tokens)`, which is a much better starting point.

Fast smoke test:

```bash
python train_torchgfn_music.py --steps 300 --batch 16 --bars 2 --hidden 128 --device cpu --out out_smoke
```

More realistic run:

```bash
python train_torchgfn_music.py --steps 3000 --batch 32 --bars 8 --hidden 256 --device cuda --out out
```
