# src/evaluate.py
import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from .baselines import run_baseline_random, run_baseline_echp, rollout
from .energy_model import EnergyParams
from .metrics import aggregate_histories
from .plots import plot_three_way_compare
from .rl_agent import QLearningAgent, QConfig
from .train import load_agent
from .wsn_env import WSNEnv, EnvConfig


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


def main():
    cfg = EnvConfig(
        n_nodes=100,
        area_w=100.0,
        area_h=100.0,
        bs_pos=(50.0, 50.0),
        init_energy=1.0,
        packet_bits=4000,
        topk_candidates=10,
        dead_ratio_terminate=1.0,
    )

    eparams = EnergyParams(
        E_elec=50e-9,
        eps_fs=10e-12,
        eps_mp=0.0013e-12,
        E_da=5e-9,
    )
    env = WSNEnv(cfg, eparams)

    qcfg = QConfig(topk_candidates=cfg.topk_candidates)
    area_diag = float(np.hypot(cfg.area_w, cfg.area_h))
    agent = QLearningAgent(qcfg=qcfg, n_nodes=cfg.n_nodes, init_energy=cfg.init_energy, area_diag=area_diag)

    repo_root = Path(__file__).resolve().parents[1]
    q_table_path = repo_root / "results" / "logs" / "q_table.json"
    if q_table_path.exists():
        load_agent(agent, str(q_table_path))
    else:
        print(f"[WARN] Q-table not found at '{q_table_path}'. Evaluating RL policy with current (untrained) agent.")

    seeds = list(range(1, 11))
    agg_random, agg_echp, agg_rl = evaluate_all(env, agent, seeds)
    plot_three_way_compare(agg_random, agg_echp, agg_rl, out_dir="results/figures")

    print("Random:", agg_random)
    print("ECHP:", agg_echp)
    print("RL:", agg_rl)

    def metric_or_na(agg: Dict[str, Any], key: str) -> str:
        val = agg.get(key)
        if val is None:
            return "N/A"
        try:
            if np.isnan(val):
                return "N/A"
        except TypeError:
            pass
        return f"{float(val):.1f}"

    print("\n===== FINAL TABLE =====")
    print(f"{'Method':<10} | {'FND':<5} | {'HND':<5} | {'LND':<5} | {'Rounds':<6}")
    print("-" * 45)
    rows = [
        ("Random", agg_random),
        ("ECHP", agg_echp),
        ("RL", agg_rl),
    ]
    for method, agg in rows:
        fnd = metric_or_na(agg, "FND_mean")
        hnd = metric_or_na(agg, "HND_mean")
        lnd = metric_or_na(agg, "LND_mean")
        rounds = metric_or_na(agg, "rounds_mean")
        print(f"{method:<10} | {fnd:<5} | {hnd:<5} | {lnd:<5} | {rounds:<6}")

    table_dir = repo_root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "final_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "FND", "HND", "LND", "Rounds"])
        for method, agg in rows:
            writer.writerow([
                method,
                metric_or_na(agg, "FND_mean"),
                metric_or_na(agg, "HND_mean"),
                metric_or_na(agg, "LND_mean"),
                metric_or_na(agg, "rounds_mean"),
            ])
    print(f"\nSaved CSV table to: {csv_path}")


if __name__ == "__main__":
    main()
