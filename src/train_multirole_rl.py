import os
import numpy as np

from src.config import EnvConfig
from src.energy_model import EnergyParams
from src.wsn_env import WSNEnv
from src.rl_multirole_agent import MultiRoleQAgent
from src.multi_role_policy import build_role_triples


def train(n_episodes=300):
    cfg = EnvConfig()
    eparams = EnergyParams()
    agent = MultiRoleQAgent()

    rewards = []

    for ep in range(1, n_episodes + 1):
        env = WSNEnv(cfg, eparams)
        env.reset(seed=ep)

        done = False
        total_reward = 0.0

        while not done:
            state, action_idx, pair = agent.select_action(env, training=True)

            if pair is None:
                break

            main_ch, deputy_ch, relay_ch = pair
            _, reward, done, info = env.step_multi_role(main_ch, deputy_ch, relay_ch)

            next_state = agent.discretize_state(env)
            next_pairs = build_role_triples(env, top_k=env.cfg.top_k_candidates)

            agent.update(
                state=state,
                action_idx=action_idx,
                reward=reward,
                next_state=next_state,
                next_num_actions=len(next_pairs),
                done=done
            )

        agent.decay_epsilon()
        rewards.append(total_reward)

        if ep % 25 == 0:
            print(f"Episode {ep}/{n_episodes}, reward={total_reward:.3f}, eps={agent.epsilon:.3f}")

    os.makedirs("results/tables", exist_ok=True)
    agent.save("results/tables/rl_mr_qtable.pkl")

    print("Saved RL-MR Q-table to results/tables/rl_mr_qtable.pkl")


if __name__ == "__main__":
    train(n_episodes=1000)