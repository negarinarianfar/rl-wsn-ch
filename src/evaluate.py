# src/evaluate.py
import numpy as np
from typing import Dict, Any, List

from .wsn_env import WSNEnv
from .baselines import run_baseline_random, run_baseline_echp, rollout
from .metrics import aggregate_histories
from .rl_agent import QLearningAgent
from .train import load_agent


def run_baseline_rl_greedy(env: WSNEnv, agent: QLearningAgent, seed: int) -> Dict[str, Any]:
    env.reset(seed=seed)

    def select_ch():
        s = env.extract_state()
        candidates = env.build_candidates()
        return agent.get_greedy_action(s, candidates)

    return rollout(env, select_ch=select_ch)


def evaluate_all(env: WSNEnv, agent: QLearningAgent, seeds: List[int]):
    random_h, echp_h, rl_h = [], [], []

    for sd in seeds:
        random_h.append(run_baseline_random(env, seed=sd))
        echp_h.append(run_baseline_echp(env, seed=sd))
        rl_h.append(run_baseline_rl_greedy(env, agent, seed=sd))

    agg_random = aggregate_histories(random_h)
    agg_echp = aggregate_histories(echp_h)
    agg_rl = aggregate_histories(rl_h)

    return agg_random, agg_echp, agg_rl