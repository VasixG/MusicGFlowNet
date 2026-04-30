"""
Tiny multi-track music GFlowNet using the torchgfn library.

This file intentionally uses torchgfn's Env/Sampler/TBGFlowNet abstractions.
The custom code is only the music environment, reward, MIDI export, and plotting.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from tqdm import tqdm

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.preprocessors import Preprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates


# -----------------------------
# Music representation
# -----------------------------

DRUM, BASS, CHORD, LEAD = 0, 1, 2, 3
TRACK_NAMES = ["drums", "bass", "chords", "lead"]

# Token ids per track. 0 is REST for every track.
REST = 0
# drums: 1 kick, 2 snare, 3 hihat, 4 kick+hihat, 5 snare+hihat
DRUM_TOKENS = {0: [], 1: [36], 2: [38], 3: [42], 4: [36, 42], 5: [38, 42]}
# bass/lead/chord tokens are interpreted diatonically in C minor-ish.
SCALE = np.array([0, 2, 3, 5, 7, 8, 10])  # natural minor degrees
ROOT = 48
BASS_TOKENS = (0, 1, 2, 3, 4, 5, 6, 7)
CHORD_TOKENS = (0, 1, 3, 4, 5, 6)
LEAD_TOKENS = (0, 1, 2, 3, 4, 5, 6, 7)
LOG_REWARD_MIN = -5.0
LOG_REWARD_MAX = 5.0
SCORE_RANGE = 2.0


@dataclass
class MusicConfig:
    bars: int = 8
    steps_per_bar: int = 16
    tracks: int = 4
    n_tokens: int = 12  # tokens 0..11; meaning depends on track
    bpm: int = 120
    reward_temperature: float = 2.0

    @property
    def seq_steps(self) -> int:
        return self.bars * self.steps_per_bar

    @property
    def n_cells(self) -> int:
        return self.seq_steps * self.tracks

    @property
    def state_dim(self) -> int:
        # first slot is write pointer, remaining slots are flattened grid cells
        return 1 + self.n_cells


def flat_index_to_step_track(idx: torch.Tensor, tracks: int) -> tuple[torch.Tensor, torch.Tensor]:
    step = idx // tracks
    track = idx % tracks
    return step, track


def action_token_masks(pos: torch.Tensor, cfg: MusicConfig) -> torch.Tensor:
    """Allow the full token vocabulary.

    Musical preferences belong in the reward. Hard action masks made a single
    repeated pattern too easy to dominate the distribution.
    """
    return torch.ones((*pos.shape, cfg.n_tokens), dtype=torch.bool, device=pos.device)


def token_degree(track: torch.Tensor) -> torch.Tensor:
    return torch.remainder(torch.clamp(track.long(), min=1) - 1, len(SCALE))


def chord_members(chord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = token_degree(chord)
    third = torch.remainder(root + 2, len(SCALE))
    fifth = torch.remainder(root + 4, len(SCALE))
    return root, third, fifth


def normalized_token_entropy(track: torch.Tensor, n_tokens: int) -> torch.Tensor:
    counts = torch.nn.functional.one_hot(track.clamp(min=0, max=n_tokens - 1), n_tokens).float().sum(dim=1)
    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
    return entropy / math.log(n_tokens)


def max_token_fraction(track: torch.Tensor, n_tokens: int) -> torch.Tensor:
    counts = torch.nn.functional.one_hot(track.clamp(min=0, max=n_tokens - 1), n_tokens).float().sum(dim=1)
    return counts.max(dim=1).values / track.shape[1]


# -----------------------------
# torchgfn preprocessor
# -----------------------------

class MusicPreprocessor(Preprocessor):
    """Flatten integer state to float features.

    state[0] = write pointer in [0, n_cells]
    state[1:] = token ids, -1 for unfilled cells
    """

    def __init__(self, cfg: MusicConfig):
        super().__init__(output_dim=cfg.state_dim)
        self.cfg = cfg

    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        x = states.tensor.float()
        # Normalize pointer and token values for easier neural learning.
        x0 = x[..., :1] / max(1, self.cfg.n_cells)
        xt = (x[..., 1:] + 1.0) / max(1, self.cfg.n_tokens)  # -1 -> 0
        return torch.cat([x0, xt], dim=-1)


# -----------------------------
# torchgfn custom environment
# -----------------------------

class MusicLoopEnv(DiscreteEnv):
    """Append-only fixed-grid music environment.

    At each nonterminal state, exactly one grid cell is filled.
    Actions are token ids 0..n_tokens-1. The last library action is EXIT.
    Terminating states are full grids with pointer == n_cells; then EXIT goes to sf.
    """

    def __init__(self, cfg: MusicConfig, device: str | torch.device = "cpu", debug: bool = False):
        self.cfg = cfg
        device = torch.device(device)
        s0 = torch.full((cfg.state_dim,), -1, dtype=torch.long, device=device)
        s0[0] = 0
        sf = torch.full((cfg.state_dim,), -999, dtype=torch.long, device=device)
        super().__init__(
            n_actions=cfg.n_tokens + 1,  # token actions + exit
            s0=s0,
            state_shape=(cfg.state_dim,),
            sf=sf,
            debug=debug,
        )

    def make_states_class(self) -> type[DiscreteStates]:
        env = self

        class MusicStates(DiscreteStates):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions

            def _compute_forward_masks(self) -> torch.Tensor:
                pos = self.tensor[..., 0]
                masks = torch.zeros((*self.batch_shape, env.n_actions), dtype=torch.bool, device=self.device)
                can_fill = (pos >= 0) & (pos < env.cfg.n_cells)
                can_exit = pos == env.cfg.n_cells
                if can_fill.any():
                    token_masks = action_token_masks(pos, env.cfg)
                    masks[..., : env.cfg.n_tokens] = token_masks & can_fill.unsqueeze(-1)
                masks[..., env.cfg.n_tokens] = can_exit
                return masks

            def _compute_backward_masks(self) -> torch.Tensor:
                # Backward policy has n_actions - 1 outputs: token ids only.
                pos = self.tensor[..., 0]
                masks = torch.zeros((*self.batch_shape, env.cfg.n_tokens), dtype=torch.bool, device=self.device)
                can_go_back = (pos > 0) & (pos <= env.cfg.n_cells)
                if can_go_back.any():
                    prev_cell = torch.clamp(pos - 1, min=0, max=env.cfg.n_cells - 1)
                    prev_token = self.tensor[..., 1:].gather(-1, prev_cell.unsqueeze(-1)).squeeze(-1)
                    valid = can_go_back & (prev_token >= 0) & (prev_token < env.cfg.n_tokens)
                    masks[valid, prev_token[valid].long()] = True
                return masks

        return MusicStates

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        x = states.tensor.clone()
        pos = x[:, 0].long()
        a = actions.tensor.squeeze(-1).long()
        rows = torch.arange(x.shape[0], device=x.device)
        x[rows, 1 + pos] = a
        x[:, 0] = pos + 1
        return self.States(x)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        x = states.tensor.clone()
        pos = x[:, 0].long()
        rows = torch.arange(x.shape[0], device=x.device)
        prev = torch.clamp(pos - 1, min=0)
        x[rows, 1 + prev] = -1
        x[:, 0] = prev
        return self.States(x)

    def make_random_states(self, batch_shape: Tuple[int, ...], conditions=None, device=None, debug=False):
        device = self.device if device is None else device
        x = torch.full((*batch_shape, self.cfg.state_dim), -1, dtype=torch.long, device=device)
        # Random full states; useful for debugging/evaluation.
        x[..., 0] = self.cfg.n_cells
        x[..., 1:] = torch.randint(0, self.cfg.n_tokens, (*batch_shape, self.cfg.n_cells), device=device)
        return self.States(x, conditions=conditions, debug=debug)

    def log_reward(self, states: DiscreteStates) -> torch.Tensor:
        return music_log_reward(states.tensor, self.cfg)


def music_log_reward(state_tensor: torch.Tensor, cfg: MusicConfig) -> torch.Tensor:
    """Positive terminal reward, returned as log reward.

    The reward defines a smooth probability landscape rather than one best loop:
    - drums get mild groove preferences, not hard constraints
    - bass gets a soft preference for current chord tones
    - repetition is useful only with enough variation
    - entropy and anti-dominance terms discourage mode collapse
    """
    B = state_tensor.shape[0]
    grid = state_tensor[:, 1:].view(B, cfg.seq_steps, cfg.tracks).long()
    device = grid.device

    steps = torch.arange(cfg.seq_steps, device=device)
    beat_pos = steps % cfg.steps_per_bar
    downbeat = beat_pos == 0
    backbeat = (beat_pos == 4) | (beat_pos == 12)
    off8 = (beat_pos % 4) == 2

    drums = grid[:, :, DRUM]
    bass = grid[:, :, BASS]
    chord = grid[:, :, CHORD]
    lead = grid[:, :, LEAD]

    # Drum groove: anchor first, density second.
    kick = (drums == 1) | (drums == 4)
    snare = (drums == 2) | (drums == 5)
    hat = (drums == 3) | (drums == 4) | (drums == 5)
    kick_score = kick[:, downbeat].float().mean(dim=1)
    snare_score = snare[:, backbeat].float().mean(dim=1)
    hat_score = hat[:, off8].float().mean(dim=1)
    drum_density = (drums != REST).float().mean(dim=1)
    drum_score = 0.35 * kick_score + 0.35 * snare_score + 0.15 * hat_score - 0.20 * (drum_density - 0.45).abs()

    # Harmony: bass should be a chord tone, especially on downbeats.
    bass_active = bass != REST
    chord_active = chord != REST
    root, third, fifth = chord_members(chord)
    bass_deg = token_degree(bass)
    bass_in_chord = (bass_deg == root) | (bass_deg == third) | (bass_deg == fifth)
    active_harmony = bass_active & chord_active
    harmony_denom = active_harmony.float().sum(dim=1).clamp_min(1.0)
    harmony_score = (bass_in_chord.float() * active_harmony.float()).sum(dim=1) / harmony_denom
    downbeat_harmony = (
        bass_in_chord.float()[:, downbeat]
        * bass_active.float()[:, downbeat]
        * chord_active.float()[:, downbeat]
    ).mean(dim=1)

    bass_density = bass_active.float().mean(dim=1)
    lead_density = (lead != REST).float().mean(dim=1)
    chord_downbeat = chord_active.float()[:, downbeat].mean(dim=1)
    bass_downbeat = bass_active.float()[:, downbeat].mean(dim=1)
    chord_score = 0.25 * chord_downbeat - 0.20 * (chord_active.float().mean(dim=1) - 0.25).abs()
    bass_score = 0.25 * harmony_score + 0.20 * downbeat_harmony + 0.15 * bass_downbeat - 0.20 * (bass_density - 0.25).abs()
    lead_score = 0.25 - 0.30 * (lead_density - 0.28).abs()

    # Repetition: bar 1 -> bar 2 should lock; later bars can vary but stay related.
    if cfg.bars >= 2:
        bars = grid.view(B, cfg.bars, cfg.steps_per_bar, cfg.tracks)
        bar1 = bars[:, 0]
        bar2 = bars[:, 1]
        repeat_all = (bar1 == bar2).float().mean(dim=(1, 2))
        repeat_rhythm = (bar1[:, :, DRUM] == bar2[:, :, DRUM]).float().mean(dim=1)
        repeat_harmony = (bar1[:, :, CHORD] == bar2[:, :, CHORD]).float().mean(dim=1)
        repetition_score = 0.20 * repeat_all + 0.20 * repeat_rhythm + 0.20 * repeat_harmony
        if cfg.bars >= 4:
            bar3 = bars[:, 2]
            bar4 = bars[:, 3]
            variation_13 = (bar1 == bar3).float().mean(dim=(1, 2))
            variation_24 = (bar2 == bar4).float().mean(dim=(1, 2))
            # Aim for recognizable variation, not a clone and not unrelated noise.
            target = torch.full_like(variation_13, 0.65)
            repetition_score = repetition_score + 0.15 * (1.0 - (variation_13 - target).abs())
            repetition_score = repetition_score + 0.10 * (1.0 - (variation_24 - target).abs())
    else:
        repetition_score = torch.zeros(B, device=device)

    track_entropies = torch.stack(
        [
            normalized_token_entropy(drums, cfg.n_tokens),
            normalized_token_entropy(bass, cfg.n_tokens),
            normalized_token_entropy(chord, cfg.n_tokens),
            normalized_token_entropy(lead, cfg.n_tokens),
        ],
        dim=1,
    )
    diversity_score = track_entropies.mean(dim=1)
    dominance = torch.stack(
        [
            max_token_fraction(drums, cfg.n_tokens),
            max_token_fraction(bass, cfg.n_tokens),
            max_token_fraction(chord, cfg.n_tokens),
            max_token_fraction(lead, cfg.n_tokens),
        ],
        dim=1,
    ).mean(dim=1)
    anti_collapse_score = diversity_score - torch.relu(dominance - 0.55)

    score = (
        1.0 * drum_score
        + 1.0 * bass_score
        + 1.0 * chord_score
        + 0.8 * lead_score
        + 0.7 * repetition_score
        + 0.6 * anti_collapse_score
    )
    if B > 1:
        score = (score - score.mean()) / score.std(unbiased=False).clamp_min(1e-6)
    score = torch.clamp(score, min=-SCORE_RANGE, max=SCORE_RANGE)

    # Return logR directly. Keeping this bounded is critical for Trajectory Balance.
    log_reward = score / cfg.reward_temperature
    return torch.clamp(log_reward, min=LOG_REWARD_MIN, max=LOG_REWARD_MAX)


# -----------------------------
# Model
# -----------------------------

class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# -----------------------------
# Decode, MIDI export, plots
# -----------------------------

def state_to_grid(state: torch.Tensor, cfg: MusicConfig) -> np.ndarray:
    arr = state.detach().cpu().long().numpy()[1:]
    return arr.reshape(cfg.seq_steps, cfg.tracks)


def token_to_pitch(track: int, token: int) -> list[int]:
    if token == REST:
        return []
    if track == DRUM:
        return DRUM_TOKENS.get(int(token), [])
    deg = int((token - 1) % len(SCALE))
    octave = int((token - 1) // len(SCALE))
    if track == BASS:
        return [36 + int(SCALE[deg]) + 12 * min(octave, 1)]
    if track == LEAD:
        return [60 + int(SCALE[deg]) + 12 * min(octave, 1)]
    if track == CHORD:
        root = 48 + int(SCALE[deg])
        return [root, root + 3, root + 7]
    return []


def export_midi(grid: np.ndarray, cfg: MusicConfig, path: str):
    pm = pretty_midi.PrettyMIDI(initial_tempo=cfg.bpm)
    seconds_per_step = 60.0 / cfg.bpm / 4.0  # 16th note
    programs = [0, 33, 0, 80]
    names = ["Drums", "Bass", "Chords", "Lead"]
    instruments = []
    for t in range(cfg.tracks):
        inst = pretty_midi.Instrument(program=programs[t], is_drum=(t == DRUM), name=names[t])
        instruments.append(inst)
        pm.instruments.append(inst)

    for step in range(cfg.seq_steps):
        start = step * seconds_per_step
        end = start + seconds_per_step * 0.95
        for track in range(cfg.tracks):
            token = int(grid[step, track])
            for pitch in token_to_pitch(track, token):
                velocity = 90 if track == DRUM else 75
                instruments[track].notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.write(path)


def plot_grid(grid: np.ndarray, cfg: MusicConfig, path: str):
    fig, axes = plt.subplots(cfg.tracks, 1, figsize=(14, 7), sharex=True)
    for tr in range(cfg.tracks):
        axes[tr].imshow(grid[:, tr][None, :], aspect="auto", interpolation="nearest")
        axes[tr].set_yticks([])
        axes[tr].set_ylabel(TRACK_NAMES[tr])
        for bar in range(cfg.bars + 1):
            axes[tr].axvline(bar * cfg.steps_per_bar - 0.5, linewidth=0.5)
    axes[-1].set_xlabel("16th-note grid step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_loss(losses: list[float], path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("TBGFlowNet loss")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def sample_best(gfn: TBGFlowNet, env: MusicLoopEnv, n: int = 64):
    states = gfn.sample_terminating_states(env, n)
    rewards = env.log_reward(states).exp()
    idx = torch.argmax(rewards)
    return states.tensor[idx], float(rewards[idx].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--logz-lr", type=float, default=5e-3)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--logz-init", type=str, default="auto", help="auto or numeric. auto≈n_cells*log(n_tokens)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = MusicConfig(bars=args.bars)
    env = MusicLoopEnv(cfg, device=args.device)
    pre = MusicPreprocessor(cfg)

    module_pf = TinyMLP(pre.output_dim, env.n_actions, hidden=args.hidden).to(args.device)
    module_pb = TinyMLP(pre.output_dim, env.n_actions - 1, hidden=args.hidden).to(args.device)

    pf = DiscretePolicyEstimator(module_pf, env.n_actions, is_backward=False, preprocessor=pre)
    pb = DiscretePolicyEstimator(module_pb, env.n_actions, is_backward=True, preprocessor=pre)
    init_logz = cfg.n_cells * math.log(cfg.n_tokens) if args.logz_init == "auto" else float(args.logz_init)
    gfn = TBGFlowNet(pf=pf, pb=pb, init_logZ=init_logz)
    print(f"init_logZ={init_logz:.2f}; n_cells={cfg.n_cells}; rough trajectory length={cfg.n_cells}")
    sampler = Sampler(estimator=pf)

    optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": args.logz_lr})

    losses = []
    best_reward = -1.0
    best_state = None

    for step in (bar := tqdm(range(args.steps))):
        traj = sampler.sample_trajectories(
            env=env,
            n=args.batch,
            save_logprobs=True,
            epsilon=args.epsilon,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = gfn.loss(env, traj)
        loss.backward()
        trainable_params = list(gfn.pf_pb_parameters()) + list(gfn.logz_parameters())
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if step % args.eval_every == 0 or step == args.steps - 1:
            st, rew = sample_best(gfn, env, n=32)
            if rew > best_reward:
                best_reward = rew
                best_state = st.detach().cpu()
            bar.set_postfix(loss=f"{loss.item():.3f}", best_R=f"{best_reward:.3f}")

    assert best_state is not None
    grid = state_to_grid(best_state, cfg)
    export_midi(grid, cfg, str(out / "best_sample.mid"))
    plot_grid(grid, cfg, str(out / "best_sample.png"))
    plot_loss(losses, str(out / "loss.png"))

    torch.save(
        {
            "model_pf": module_pf.state_dict(),
            "model_pb": module_pb.state_dict(),
            "cfg": cfg.__dict__,
            "best_reward": best_reward,
        },
        out / "model.pt",
    )
    print(f"Saved: {out / 'best_sample.mid'}")
    print(f"Saved: {out / 'best_sample.png'}")
    print(f"Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
