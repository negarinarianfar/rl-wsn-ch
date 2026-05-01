import numpy as np


def alive_nodes(env):
    return np.where(env.energy > 0)[0]


def select_deputy_high_energy(env, main_ch):
    alive = alive_nodes(env)
    candidates = [i for i in alive if i != main_ch]

    if not candidates:
        return None

    return int(max(candidates, key=lambda i: env.energy[i]))


def select_deputy_balanced(env, main_ch):
    alive = alive_nodes(env)
    candidates = [i for i in alive if i != main_ch]

    if not candidates:
        return None

    bs = np.array(env.cfg.bs_pos)

    def score(i):
        energy_score = env.energy[i] / (np.max(env.energy) + 1e-9)
        dist_to_bs = np.linalg.norm(env.positions[i] - bs)
        dist_score = 1.0 / (dist_to_bs + 1e-9)
        return 0.7 * energy_score + 0.3 * dist_score

    return int(max(candidates, key=score))


def build_role_pairs(env, top_k=10):
    alive = alive_nodes(env)

    if len(alive) < 2:
        return []

    top_main = sorted(alive, key=lambda i: env.energy[i], reverse=True)[:top_k]

    pairs = []
    for main_ch in top_main:
        deputy = select_deputy_balanced(env, main_ch)
        if deputy is not None:
            pairs.append((int(main_ch), int(deputy)))

    return pairs