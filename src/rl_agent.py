# src/rl_agent.py
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict


@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 300  # linear decay
    topk_candidates: int = 10

    # discretization bins (counts)
    bins_avgE: int = 10
    bins_minE: int = 10
    bins_varE: int = 10
    bins_dist: int = 10
    bins_age: int = 5


class QLearningAgent:
    """
    Q-table stored as dict: Q[(state_bin_tuple, action_node_id)] -> value
    This avoids fixing N at compile time and is easy to implement.
    """
    def __init__(self, qcfg: QConfig, n_nodes: int, init_energy: float, area_diag: float):
        self.qcfg = qcfg
        self.n_nodes = n_nodes

        # For discretization ranges (simple, practical defaults)
        # Energy roughly in [0, init_energy]
        self.e_max = float(init_energy)
        self.var_max = float(init_energy ** 2)  # conservative
        self.dist_max = float(area_diag)        # max possible distance in area
        self.age_max = 50.0                     # cap; can tune later

        self.Q: Dict[Tuple[Tuple[int, int, int, int, int], int], float] = {}

        self.rng = np.random.default_rng(0)

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def epsilon(self, episode_idx: int) -> float:
        # linear decay
        if episode_idx >= self.qcfg.eps_decay_episodes:
            return self.qcfg.eps_end
        frac = episode_idx / float(self.qcfg.eps_decay_episodes)
        return self.qcfg.eps_start + frac * (self.qcfg.eps_end - self.qcfg.eps_start)

    def _bin(self, x: float, x_max: float, n_bins: int) -> int:
        if n_bins <= 1:
            return 0
        x = max(0.0, min(x, x_max))
        # map [0, x_max] -> {0..n_bins-1}
        return int(np.floor((x / x_max) * (n_bins - 1))) if x_max > 0 else 0

    def discretize_state(self, s: np.ndarray) -> Tuple[int, int, int, int, int]:
        avgE, minE, varE, distPrev, age = [float(v) for v in s]
        b1 = self._bin(avgE, self.e_max, self.qcfg.bins_avgE)
        b2 = self._bin(minE, self.e_max, self.qcfg.bins_minE)
        b3 = self._bin(varE, self.var_max, self.qcfg.bins_varE)
        b4 = self._bin(distPrev, self.dist_max, self.qcfg.bins_dist)
        b5 = self._bin(age, self.age_max, self.qcfg.bins_age)
        return (b1, b2, b3, b4, b5)

    def q_value(self, s_bin: Tuple[int, int, int, int, int], a: int) -> float:
        return self.Q.get((s_bin, int(a)), 0.0)

    def choose_action(self, s: np.ndarray, candidates: np.ndarray, episode_idx: int) -> int:
        eps = self.epsilon(episode_idx)
        if candidates.size == 0:
            return 0

        if self.rng.random() < eps:
            return int(self.rng.choice(candidates))

        s_bin = self.discretize_state(s)
        # argmax Q over candidates
        q_vals = [self.q_value(s_bin, a) for a in candidates]
        best_idx = int(np.argmax(q_vals))
        return int(candidates[best_idx])

    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, next_candidates: np.ndarray):
        s_bin = self.discretize_state(s)
        a = int(a)

        q_sa = self.q_value(s_bin, a)

        if next_candidates.size == 0:
            target = r
        else:
            s2_bin = self.discretize_state(s_next)
            q_next = max(self.q_value(s2_bin, a2) for a2 in next_candidates)
            target = r + self.qcfg.gamma * q_next

        new_q = q_sa + self.qcfg.alpha * (target - q_sa)
        self.Q[(s_bin, a)] = float(new_q)

    def get_greedy_action(self, s: np.ndarray, candidates: np.ndarray) -> int:
        if candidates.size == 0:
            return 0
        s_bin = self.discretize_state(s)
        q_vals = [self.q_value(s_bin, a) for a in candidates]
        return int(candidates[int(np.argmax(q_vals))])