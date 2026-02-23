# src/energy_model.py
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EnergyParams:
    E_elec: float      # J/bit
    eps_fs: float      # J/bit/m^2
    eps_mp: float      # J/bit/m^4
    E_da: float = 0.0  # J/bit (optional data aggregation cost)


def d0(params: EnergyParams) -> float:
    return math.sqrt(params.eps_fs / params.eps_mp)


def e_rx(k_bits: int, params: EnergyParams) -> float:
    return k_bits * params.E_elec


def e_tx(k_bits: int, dist: float, params: EnergyParams) -> float:
    # Standard radio model: free-space (d^2) vs multipath (d^4)
    if dist < d0(params):
        amp = params.eps_fs * (dist ** 2)
    else:
        amp = params.eps_mp * (dist ** 4)
    return k_bits * params.E_elec + k_bits * amp


def e_da(k_bits: int, params: EnergyParams) -> float:
    return k_bits * params.E_da