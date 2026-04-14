# src/train.py
import os
import json
import numpy as np
from typing import Dict, Any

from .wsn_env import WSNEnv
from .rl_agent import QLearningAgent


def train_qlearning(
    env: WSNEnv,
    agent: QLearningAgent,
    episodes: int = 300,
    max_rounds_per_episode: int | None = None,
    seed: int = 0,
) -> Dict[str, Any]:
    agent.set_seed(seed)

    rewards = []
    lengths = []

    for ep in range(episodes):
        s = env.reset(seed=seed + ep + 1)

        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            candidates = env.build_candidates()
            a = agent.choose_action(s, candidates, episode_idx=ep)

            s_next, r, done, info = env.step(a)
            next_candidates = env.build_candidates()

            agent.update(s, a, r, s_next, next_candidates)

            s = s_next
            ep_reward += r
            steps += 1

            if max_rounds_per_episode is not None and steps >= max_rounds_per_episode:
                break

        rewards.append(ep_reward)
        lengths.append(steps)

        if (ep + 1) % 25 == 0:
            print(f"Episode {ep+1}/{episodes}: reward={ep_reward:.3f}, steps={steps}, eps={agent.epsilon(ep):.3f}")

    return {
        "episode_rewards": np.array(rewards, dtype=float),
        "episode_lengths": np.array(lengths, dtype=int),
    }


def save_agent(agent: QLearningAgent, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # save Q dict as json-friendly list
    serial = [{"s": list(k[0]), "a": int(k[1]), "q": float(v)} for k, v in agent.Q.items()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serial, f)


def load_agent(agent: QLearningAgent, path: str):
    import json
    with open(path, "r", encoding="utf-8") as f:
        serial = json.load(f)
    agent.Q.clear()
    for item in serial:
        s_bin = tuple(int(x) for x in item["s"])
        a = int(item["a"])
        q = float(item["q"])
        agent.Q[(s_bin, a)] = q