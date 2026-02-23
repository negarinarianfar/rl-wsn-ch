# src/energy_model.py
from dataclasses import dataclass
import math

@dataclass
class EnergyParams:
    """
    First-order radio energy model parameters (common in WSN/LEACH-like papers).
    Units:
      - Energies: Joule
      - Distances: meter
      - L: bits
    """
    E_elec: float = 50e-9        # J/bit
    eps_fs: float = 10e-12       # J/bit/m^2  (free space)
    eps_mp: float = 0.0013e-12   # J/bit/m^4  (multi-path)
    E_da: float = 5e-9           # J/bit (data aggregation)

    @property
    def d0(self) -> float:
        # threshold distance where free-space switches to multi-path
        return math.sqrt(self.eps_fs / self.eps_mp)


def tx_energy(L_bits: int, d: float, p: EnergyParams) -> float:
    """Energy to transmit L bits over distance d."""
    if d < p.d0:
        return L_bits * (p.E_elec + p.eps_fs * (d ** 2))
    return L_bits * (p.E_elec + p.eps_mp * (d ** 4))


def rx_energy(L_bits: int, p: EnergyParams) -> float:
    """Energy to receive L bits."""
    return L_bits * p.E_elec


def da_energy(L_bits: int, p: EnergyParams) -> float:
    """Energy for data aggregation for L bits."""
    return L_bits * p.E_da


# --- Wrappers compatible with wsn_env.py (which passes kbits + params) ---

def tx_energy_kbits(kbits: float, d: float, p: EnergyParams) -> float:
    """Transmit energy for kbits over distance d."""
    L_bits = int(kbits * 1000)
    return tx_energy(L_bits, d, p)

def rx_energy_kbits(kbits: float, p: EnergyParams) -> float:
    """Receive energy for kbits."""
    L_bits = int(kbits * 1000)
    return rx_energy(L_bits, p)

def da_energy_kbits(kbits: float, p: EnergyParams) -> float:
    """Data aggregation energy for kbits."""
    L_bits = int(kbits * 1000)
    return da_energy(L_bits, p)