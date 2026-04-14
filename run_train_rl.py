# run_train_rl.py
import os
import numpy as np
import matplotlib.pyplot as plt

from src.energy_model import EnergyParams
from src.wsn_env import EnvConfig, WSNEnv
from src.rl_agent import QConfig, QLearningAgent
from src.train import train_qlearning, save_agent


def main():
    cfg = EnvConfig(
        n_nodes=100,
        area_w=100.0,
        area_h=100.0,
        bs_pos=(50.0, 50.0),
        init_energy=1.0,
        packet_bits=4000,
        topk_candidates=10,
        dead_ratio_terminate=0.8,
    )

    eparams = EnergyParams(
        E_elec=50e-9,
        eps_fs=10e-12,
        eps_mp=0.0013e-12,
        E_da=5e-9,
    )

    env = WSNEnv(cfg, eparams)

    qcfg = QConfig(
        alpha=0.1,
        gamma=0.9,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=300,
        topk_candidates=cfg.topk_candidates,
    )

    area_diag = (cfg.area_w**2 + cfg.area_h**2) ** 0.5
    agent = QLearningAgent(qcfg, n_nodes=cfg.n_nodes, init_energy=cfg.init_energy, area_diag=area_diag)

    logs = train_qlearning(env, agent, episodes=300, max_rounds_per_episode=None, seed=42)

    # save Q-table
    os.makedirs("results/logs", exist_ok=True)
    save_agent(agent, "results/logs/q_table.json")

    # plot reward curve
    os.makedirs("results/figures", exist_ok=True)
    rewards = logs["episode_rewards"]
    plt.figure()
    plt.plot(np.arange(1, len(rewards) + 1), rewards)
    plt.title("Training Reward per Episode (Q-learning)")
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.tight_layout()
    plt.savefig("results/figures/reward_train.png", dpi=200)
    plt.close()

    print("Saved Q-table to results/logs/q_table.json")
    print("Saved reward plot to results/figures/reward_train.png")


if __name__ == "__main__":
    main()