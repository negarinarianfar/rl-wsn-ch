# run_baselines.py
from src.energy_model import EnergyParams
from src.wsn_env import EnvConfig, WSNEnv
from src.baselines import run_baseline_random, run_baseline_echp
from src.metrics import aggregate_histories
from src.plots import plot_baselines


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

    seeds = list(range(1, 11))  # 10 runs
    random_h = []
    echp_h = []

    for sd in seeds:
        random_h.append(run_baseline_random(env, seed=sd))
        echp_h.append(run_baseline_echp(env, seed=sd))

    agg_random = aggregate_histories(random_h)
    agg_echp = aggregate_histories(echp_h)

    print("=== Baseline summary (mean ± std) ===")
    for name, agg in [("Random", agg_random), ("ECHP", agg_echp)]:
        print(f"\n{name}:")
        print(f"  FND: {agg['FND_mean']:.1f} ± {agg['FND_std']:.1f}")
        print(f"  HND: {agg['HND_mean']:.1f} ± {agg['HND_std']:.1f}")
        print(f"  LND: {agg['LND_mean']:.1f} ± {agg['LND_std']:.1f}")
        print(f"  Rounds: {agg['rounds_mean']:.1f} ± {agg['rounds_std']:.1f}")

    plot_baselines(agg_random, agg_echp, out_dir="results/figures")
    print("\nSaved figures to results/figures")


if __name__ == "__main__":
    main()