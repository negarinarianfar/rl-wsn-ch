import numpy as np

from src.config import EnvConfig
from src.energy_model import EnergyParams
from src.wsn_env import WSNEnv
from src.baselines import random_multi_role, leach_baseline, heed_baseline


def run_one(seed=1, policy_fn=None):
    cfg = EnvConfig()
    eparams = EnergyParams()
    env = WSNEnv(cfg, eparams)

    env.reset(seed=seed)

    done = False
    info = {}

    while not done:
        main_ch, deputy_ch = policy_fn(env)

        if main_ch is None or deputy_ch is None:
            break

        _, reward, done, info = env.step_multi_role(main_ch, deputy_ch)

    return info


def run_many(policy_fn, n_runs=10):
    FNDs, HNDs, LNDs, rounds = [], [], [], []

    for seed in range(n_runs):
        info = run_one(seed=seed, policy_fn=policy_fn)

        FNDs.append(info.get("FND"))
        HNDs.append(info.get("HND"))
        LNDs.append(info.get("LND") if info.get("LND") is not None else np.nan)
        rounds.append(info.get("round"))

    def summary(arr):
        arr = np.array(arr, dtype=float)

        if np.all(np.isnan(arr)):
            return "N/A"

        return round(float(np.nanmean(arr)), 2), round(float(np.nanstd(arr)), 2)

    print("FND:", summary(FNDs))
    print("HND:", summary(HNDs))
    print("LND:", summary(LNDs))
    print("Rounds:", summary(rounds))


if __name__ == "__main__":
    methods = {
        "Random Multi-Role": random_multi_role,
        "LEACH-MR": leach_baseline,
        "HEED-MR": heed_baseline,
    }

    for name, policy_fn in methods.items():
        print(f"\n=== {name} ===")
        run_many(policy_fn=policy_fn, n_runs=10)