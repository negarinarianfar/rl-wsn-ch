# src/metrics.py
import numpy as np
from typing import Dict, Any, List


def pad_to_length(arr: np.ndarray, L: int) -> np.ndarray:
    """Pad array to length L by repeating last value (or zeros if empty)."""
    if arr.size == 0:
        return np.zeros(L, dtype=arr.dtype if arr.dtype != object else float)
    if arr.size >= L:
        return arr[:L]
    pad_val = arr[-1]
    pad = np.full(L - arr.size, pad_val, dtype=arr.dtype)
    return np.concatenate([arr, pad])


def aggregate_histories(histories: List[Dict[str, Any]], keys=("alive", "avg_energy", "var_energy")) -> Dict[str, Any]:
    """
    Aggregate multiple rollouts:
    - Align lengths by padding with last value
    - Return mean and std curves
    - Return mean/std for FND/HND/LND and total rounds
    """
    max_len = max(h["alive"].size for h in histories) if histories else 0

    agg = {"max_len": max_len, "n_runs": len(histories)}

    for k in keys:
        mats = np.stack([pad_to_length(h[k], max_len) for h in histories], axis=0)
        agg[f"{k}_mean"] = mats.mean(axis=0)
        agg[f"{k}_std"] = mats.std(axis=0)

    # Scalars
    for sk in ("FND", "HND", "LND", "rounds"):
        vals = np.array([h[sk] if h[sk] is not None else np.nan for h in histories], dtype=float)
        agg[f"{sk}_mean"] = float(np.nanmean(vals))
        agg[f"{sk}_std"] = float(np.nanstd(vals))

    return agg