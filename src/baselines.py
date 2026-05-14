# src/baselines.py

import numpy as np
from typing import Callable, List

from .wsn_env import WSNEnv
from .multi_role_policy import build_role_pairs


# =========================
# Single-Role Baselines
# =========================

def policy_random(env: WSNEnv, rng: np.random.Generator) -> int:
    alive_idx = np.where(env.alive)[0]

    if len(alive_idx) == 0:
        return -1

    return int(rng.choice(alive_idx))


def policy_echp(env: WSNEnv) -> int:
    """
    ECHP-style heuristic:
    Select the alive node with the highest residual energy as cluster head.
    """
    alive_idx = np.where(env.alive)[0]

    if len(alive_idx) == 0:
        return -1

    energies = env.energy[alive_idx]
    return int(alive_idx[int(np.argmax(energies))])


def rollout(
    env: WSNEnv,
    select_ch: Callable[[], int],
    max_rounds: int | None = None
):
    """
    Runs one episode until the environment is done or max_rounds is reached.
    Returns per-round history and final lifetime milestones.
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

        if ch is None or ch < 0:
            break

        _, reward, done, info = env.step(ch)
        info_last = info

        alive_hist.append(info["alive"])
        cons_hist.append(info["energy_consumed"])
        r_hist.append(reward)

        if env.alive.any():
            avgE_hist.append(float(env.energy[env.alive].mean()))
            varE_hist.append(float(env.energy[env.alive].var()))
        else:
            avgE_hist.append(0.0)
            varE_hist.append(0.0)

        if max_rounds is not None and info["round"] >= max_rounds:
            break

    return {
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


def run_baseline_random(env: WSNEnv, seed: int):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    return rollout(
        env,
        select_ch=lambda: policy_random(env, rng)
    )


def run_baseline_echp(env: WSNEnv, seed: int):
    env.reset(seed=seed)

    return rollout(
        env,
        select_ch=lambda: policy_echp(env)
    )


# =========================
# Multi-Role Baselines
# =========================

def random_multi_role(env):
    """
    Random multi-role baseline.
    Uses generated role pairs and selects the first available pair.
    """

    pairs = build_role_pairs(env, top_k=env.cfg.top_k_candidates)

    if not pairs:
        return None, None

    return pairs[0]


def leach_baseline(env, p: float = 0.05):
    """
    LEACH-style probabilistic baseline.
    Selects cluster heads randomly with probability p.
    Deputy is selected randomly from alive nodes excluding the main CH.
    """

    alive = np.where(env.energy > 0)[0]

    if len(alive) == 0:
        return None, None

    if len(alive) == 1:
        node = int(alive[0])
        return node, node

    chs = [int(n) for n in alive if np.random.rand() < p]

    if len(chs) == 0:
        chs = [int(np.random.choice(alive))]

    main_ch = chs[0]

    others = [int(n) for n in alive if n != main_ch]

    if not others:
        return main_ch, main_ch

    deputy = int(np.random.choice(others))

    return main_ch, deputy


def heed_baseline(env, c_prob: float = 0.05):
    """
    HEED-style energy-aware baseline.
    CH probability is proportional to residual energy.
    Deputy is the highest-energy alive node excluding the selected CH.
    """

    alive = np.where(env.energy > 0)[0]

    if len(alive) == 0:
        return None, None

    if len(alive) == 1:
        node = int(alive[0])
        return node, node

    energies = env.energy[alive]
    max_energy = np.max(energies)

    if max_energy <= 0:
        return None, None

    probs = (energies / max_energy) * c_prob

    ch_candidates = [
        int(node)
        for node, prob in zip(alive, probs)
        if np.random.rand() < prob
    ]

    if len(ch_candidates) == 0:
        ch_candidates = [int(alive[np.argmax(energies)])]

    main_ch = ch_candidates[0]

    others = [int(n) for n in alive if n != main_ch]

    if not others:
        return main_ch, main_ch

    deputy = max(others, key=lambda n: env.energy[n])

    return main_ch, int(deputy)

from src.rl_multirole_agent import MultiRoleQAgent


_rl_agent_cache = None


def rl_multi_role_policy(env):
    global _rl_agent_cache

    if _rl_agent_cache is None:
        _rl_agent_cache = MultiRoleQAgent()
        _rl_agent_cache.load("results/tables/rl_mr_qtable.pkl")

    state, action_idx, pair = _rl_agent_cache.select_action(env, training=False)

    if pair is None:
        return None, None

    return pair