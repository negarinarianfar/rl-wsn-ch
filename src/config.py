from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    n_nodes: int = 100
    area_size: float = 100.0
    area_w: float = 100.0
    area_h: float = 100.0
    init_energy: float = 0.5
    packet_bits: int = 4000
    max_rounds: int = 2500
    top_k_candidates: int = 10
    dead_ratio_terminate: float = 0.8
    bs_pos: Tuple[float, float] = (50.0, 50.0)