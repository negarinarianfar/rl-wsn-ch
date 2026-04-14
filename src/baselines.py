# src/baselines.py
import numpy as np
from typing import Callable, Dict, Any, List, Tuple

from .wsn_env import WSNEnv


def policy_random(env: WSNEnv, rng: np.random.Generator) -> int:
    alive_idx = np.where(env.alive)[0]
    return int(rng.choice(alive_idx))


def policy_echp(env: WSNEnv) -> int:
    # ECHP-style heuristic for CH: pick alive node with max residual energy
    alive_idx = np.where(env.alive)[0]
    energies = env.energy[alive_idx]
    return int(alive_idx[int(np.argmax(energies))])


def rollout(env: WSNEnv, select_ch: Callable[[], int], max_rounds: int | None = None):
    """
    Runs one episode until done (or max_rounds if provided).
    Returns history dict with per-round arrays + final milestones.
    """
    alive_hist: List[int] = []
    avgE_hist: List[float] = []
    varE_hist: List[float] = []
    cons_hist: List[float] = []
    r_hist: List[float] = []

    done = False
    info_last = None

    while not done:
        ch = select_ch()
        s_next, r, done, info = env.step(ch)
        info_last = info

        alive_hist.append(info["alive"])
        cons_hist.append(info["energy_consumed"])
        r_hist.append(r)

        if env.alive.any():
            avgE_hist.append(float(env.energy[env.alive].mean()))
            varE_hist.append(float(env.energy[env.alive].var()))
        else:
            avgE_hist.append(0.0)
            varE_hist.append(0.0)

        if max_rounds is not None and info["round"] >= max_rounds:
            break

    out = {
        "alive": np.array(alive_hist, dtype=int),
        "avg_energy": np.array(avgE_hist, dtype=float),
        "var_energy": np.array(varE_hist, dtype=float),
        "energy_consumed": np.array(cons_hist, dtype=float),
        "reward": np.array(r_hist, dtype=float),
        "FND": info_last["FND"] if info_last else None,
        "HND": info_last["HND"] if info_last else None,
        "LND": info_last["LND"] if info_last else None,
        "rounds": info_last["round"] if info_last else 0,
    }
    return out


def run_baseline_random(env: WSNEnv, seed: int):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    return rollout(env, select_ch=lambda: policy_random(env, rng))


def run_baseline_echp(env: WSNEnv, seed: int):
    env.reset(seed=seed)
    return rollout(env, select_ch=lambda: policy_echp(env))