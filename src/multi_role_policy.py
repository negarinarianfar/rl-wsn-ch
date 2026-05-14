import numpy as np


def alive_nodes(env):
    return np.where(env.energy > 0)[0]


def select_deputy_high_energy(env, main_ch):
    alive = alive_nodes(env)
    candidates = [int(i) for i in alive if int(i) != int(main_ch)]

    if not candidates:
        return None

    return int(max(candidates, key=lambda i: env.energy[i]))


def select_deputy_balanced(env, main_ch):
    alive = alive_nodes(env)
    candidates = [i for i in alive if i != main_ch]

    if not candidates:
        return None

    bs = np.array(env.cfg.bs_pos)
    max_energy = float(np.max(env.energy)) + 1e-9

    best_node = None
    best_score = -1e18

    for i in candidates:
        energy_score = env.energy[i] / max_energy
        dist_to_bs = np.linalg.norm(env.positions[i] - bs)
        dist_score = 1.0 / (dist_to_bs + 1e-9)
        score = 0.7 * energy_score + 0.3 * dist_score

        if score > best_score:
            best_score = score
            best_node = int(i)

    return best_node


def build_role_pairs(env, top_k=10):
    alive = alive_nodes(env)

    if len(alive) < 2:
        return []

    top_main = sorted(
        alive,
        key=lambda i: env.energy[i],
        reverse=True
    )[:top_k]

    pairs = []

    for main_ch in top_main:
        deputy = select_deputy_balanced(env, main_ch)

        if deputy is not None:
            pairs.append((int(main_ch), int(deputy)))

    return pairs

def select_relay_balanced(env, main_ch, deputy_ch):
    alive = alive_nodes(env)
    candidates = [i for i in alive if i not in [main_ch, deputy_ch]]

    if not candidates:
        return None

    bs = np.array(env.cfg.bs_pos)
    max_energy = float(np.max(env.energy)) + 1e-9

    def score(i):
        energy_score = env.energy[i] / max_energy
        dist_to_bs = np.linalg.norm(env.positions[i] - bs)
        dist_score = 1.0 / (dist_to_bs + 1e-9)
        return 0.6 * energy_score + 0.4 * dist_score

    return int(max(candidates, key=score))


def build_role_triples(env, top_k=10):
    pairs = build_role_pairs(env, top_k=top_k)
    triples = []

    for main_ch, deputy_ch in pairs:
        relay_ch = select_relay_balanced(env, main_ch, deputy_ch)
        if relay_ch is not None:
            triples.append((main_ch, deputy_ch, relay_ch))

    return triples