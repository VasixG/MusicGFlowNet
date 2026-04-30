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
