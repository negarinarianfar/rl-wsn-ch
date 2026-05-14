import numpy as np
import os
import matplotlib.pyplot as plt

from src.baselines import (
    random_multi_role,
    leach_baseline,
    heed_baseline,
    rl_multi_role_policy,
)

from src.config import EnvConfig
from src.energy_model import EnergyParams
from src.wsn_env import WSNEnv


cfg = EnvConfig(
    n_nodes=100,
    area_w=100,
    area_h=100,
    init_energy=0.5,
    packet_bits=4000,
    max_rounds=2500,
    top_k_candidates=10,
    dead_ratio_terminate=0.8,
)

eparams = EnergyParams()


def run_policy(policy_fn, seed=1):
    env = WSNEnv(cfg, eparams)
    env.reset(seed=seed)

    alive_curve = []
    energy_curve = []
    dead_curve = []

    done = False

    while not done:
        result = policy_fn(env)

        if result is None:
            break

        if len(result) == 2:
            main_ch, deputy_ch = result
            relay_ch = None
        else:
            main_ch, deputy_ch, relay_ch = result

        _, _, done, info = env.step_multi_role(main_ch, deputy_ch, relay_ch)

        alive = info["alive"]
        alive_curve.append(alive)
        dead_curve.append(env.cfg.n_nodes - alive)
        energy_curve.append(float(np.mean(env.energy)))

    return {
        "alive": alive_curve,
        "dead": dead_curve,
        "energy": energy_curve,
    }


def pad_curve(curve, target_len):
    if len(curve) < target_len:
        curve = curve + [curve[-1]] * (target_len - len(curve))
    return curve


policies = {
    "Random-MR": random_multi_role,
    "LEACH-MR": leach_baseline,
    "HEED-MR": heed_baseline,
    "RL-MR": rl_multi_role_policy,
}

results = {}
max_len = 0

for name, policy in policies.items():
    curves = run_policy(policy)
    results[name] = curves
    max_len = max(max_len, len(curves["alive"]))


def pad_curve(curve, target_len):
    if len(curve) < target_len:
        curve = curve + [curve[-1]] * (target_len - len(curve))
    return curve


def plot_metric(metric_key, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    for name, curves in results.items():
        curve = pad_curve(curves[metric_key], max_len)
        plt.plot(curve, label=name)

    plt.xlabel("Rounds")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/{filename}", dpi=300)
    plt.show()


plot_metric(
    metric_key="alive",
    ylabel="Alive Nodes",
    title="Alive Nodes vs Rounds",
    filename="alive_nodes_multirole.png",
)

plot_metric(
    metric_key="energy",
    ylabel="Average Residual Energy",
    title="Average Residual Energy vs Rounds",
    filename="residual_energy_multirole.png",
)

plot_metric(
    metric_key="dead",
    ylabel="Dead Nodes",
    title="Dead Nodes vs Rounds",
    filename="dead_nodes_multirole.png",
)