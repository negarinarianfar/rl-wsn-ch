# src/wsn_env.py
from dataclasses import dataclass
import numpy as np

from .energy_model import EnergyParams, tx_energy, rx_energy, da_energy


@dataclass
class EnvConfig:
    n_nodes: int
    area_w: float
    area_h: float
    bs_pos: tuple[float, float]
    init_energy: float
    packet_bits: int
    topk_candidates: int

    # termination
    dead_ratio_terminate: float = 1.0  # episode ends when all nodes are dead


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class WSNEnv:
    def __init__(self, cfg: EnvConfig, eparams: EnergyParams):
        self.cfg = cfg
        self.eparams = eparams

        self.rng = np.random.default_rng(0)
        self.positions = None
        self.energy = None
        self.alive = None

        self.round_idx = 0
        self.prev_ch = None
        self.ch_age = 0

        self.bs = np.array(cfg.bs_pos, dtype=np.float64)

        # trackers
        self.first_dead_round = None
        self.half_dead_round = None
        self.last_dead_round = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # positions uniform
        xs = self.rng.uniform(0, self.cfg.area_w, size=self.cfg.n_nodes)
        ys = self.rng.uniform(0, self.cfg.area_h, size=self.cfg.n_nodes)
        self.positions = np.stack([xs, ys], axis=1).astype(np.float64)

        self.energy = np.full(self.cfg.n_nodes, self.cfg.init_energy, dtype=np.float64)
        self.alive = np.ones(self.cfg.n_nodes, dtype=bool)

        self.round_idx = 0
        self.prev_ch = None
        self.ch_age = 0

        self.first_dead_round = None
        self.half_dead_round = None
        self.last_dead_round = None

        return self.extract_state()

    def extract_state(self):
        # State: [avgE, minE, varE, dist_prevCH_to_BS, ch_age]
        alive_energy = self.energy[self.alive]
        avg_e = float(alive_energy.mean()) if alive_energy.size else 0.0
        min_e = float(alive_energy.min()) if alive_energy.size else 0.0
        var_e = float(alive_energy.var()) if alive_energy.size else 0.0

        if self.prev_ch is None:
            dist_prev = 0.0
        else:
            dist_prev = _dist(self.positions[self.prev_ch], self.bs)

        return np.array([avg_e, min_e, var_e, dist_prev, float(self.ch_age)], dtype=np.float64)

    def build_candidates(self):
        # top-k by energy among alive nodes
        alive_idx = np.where(self.alive)[0]
        if alive_idx.size == 0:
            return np.array([], dtype=int)

        energies = self.energy[alive_idx]
        k = min(self.cfg.topk_candidates, alive_idx.size)
        topk_local = np.argpartition(-energies, kth=k-1)[:k]
        return alive_idx[topk_local]

    def select_deputy(self, ch: int) -> int:
        # deputy = highest energy among alive excluding CH
        candidates = np.where(self.alive & (np.arange(self.cfg.n_nodes) != ch))[0]
        if candidates.size == 0:
            return ch  # degenerate: only one node alive
        best = candidates[np.argmax(self.energy[candidates])]
        return int(best)

    def _apply_energy(self, node: int, cost: float):
        if not self.alive[node]:
            return
        self.energy[node] -= cost
        if self.energy[node] <= 0.0:
            self.energy[node] = 0.0
            self.alive[node] = False

    def execute_round(self, ch: int, deputy: int):
        packet_bits = self.cfg.packet_bits
        total_consumed = 0.0
        n_member_packets_rx = 0

        # 1) members -> CH (all alive except CH and deputy)
        for i in range(self.cfg.n_nodes):
            if not self.alive[ch]:
                break
            if not self.alive[i]:
                continue
            if i == ch or i == deputy:
                continue

            d_i_ch = _dist(self.positions[i], self.positions[ch])
            tx = tx_energy(packet_bits, d_i_ch, self.eparams)
            rx = rx_energy(packet_bits, self.eparams)

            self._apply_energy(i, tx)
            self._apply_energy(ch, rx)

            total_consumed += tx + rx
            n_member_packets_rx += 1

        # 2) data aggregation at CH (optional)
        # aggregate all packets received by CH in this round
        if self.alive[ch] and n_member_packets_rx > 0:
            da_bits = packet_bits * n_member_packets_rx
            da_cost = da_energy(da_bits, self.eparams)
            self._apply_energy(ch, da_cost)
            total_consumed += da_cost

        # 3) CH -> Deputy
        if self.alive[ch] and self.alive[deputy]:
            d_ch_dep = _dist(self.positions[ch], self.positions[deputy])
            tx_ch = tx_energy(packet_bits, d_ch_dep, self.eparams)
            rx_dep = rx_energy(packet_bits, self.eparams)

            self._apply_energy(ch, tx_ch)
            self._apply_energy(deputy, rx_dep)

            total_consumed += tx_ch + rx_dep

        # 4) Deputy -> BS
        if self.alive[deputy]:
            d_dep_bs = _dist(self.positions[deputy], self.bs)
            tx_dep = tx_energy(packet_bits, d_dep_bs, self.eparams)
            self._apply_energy(deputy, tx_dep)
            total_consumed += tx_dep

        return total_consumed

    def _update_death_milestones(self):
        dead = (~self.alive).sum()
        n = self.cfg.n_nodes

        if self.first_dead_round is None and dead >= 1:
            self.first_dead_round = self.round_idx
        if self.half_dead_round is None and dead >= n / 2:
            self.half_dead_round = self.round_idx
        if dead == n and self.last_dead_round is None:
            self.last_dead_round = self.round_idx

    def compute_reward(self, energy_consumed_this_round: float):
        # Simple practical reward (can refine later):
        # + alive_ratio, - energy_consumed, - variance
        alive_ratio = float(self.alive.sum()) / float(self.cfg.n_nodes)
        var_e = float(self.energy[self.alive].var()) if self.alive.any() else 0.0

        # weights (tune later)
        lam = 1.0  # energy penalty weight
        mu = 0.1   # variance penalty weight

        r = alive_ratio - lam * energy_consumed_this_round - mu * var_e
        return float(r)

    def step(self, action_ch: int):
        if not self.alive[action_ch]:
            # invalid action -> strong penalty and end
            return self.extract_state(), -1e3, True, {"reason": "dead_CH"}

        deputy = self.select_deputy(action_ch)

        # CH age tracking
        if self.prev_ch == action_ch:
            self.ch_age += 1
        else:
            self.ch_age = 1

        consumed = self.execute_round(action_ch, deputy)

        self.round_idx += 1
        self.prev_ch = int(action_ch)

        self._update_death_milestones()

        # termination
        dead_ratio = 1.0 - (float(self.alive.sum()) / float(self.cfg.n_nodes))
        done = not self.alive.any()

        reward = self.compute_reward(consumed)
        next_state = self.extract_state()

        info = {
            "round": self.round_idx,
            "CH": int(action_ch),
            "deputy": int(deputy),
            "energy_consumed": float(consumed),
            "alive": int(self.alive.sum()),
            "FND": self.first_dead_round,
            "HND": self.half_dead_round,
            "LND": self.last_dead_round,
        }
        return next_state, reward, done, info
