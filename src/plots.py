# src/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x, y_mean, y_std, title, xlabel, ylabel, save_path):
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # تبدیل به آرایه و هم‌طول کردن
    x = np.asarray(x)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std  = np.asarray(y_std, dtype=float)

    n = min(len(x), len(y_mean), len(y_std))
    if n == 0:
        print(f"[WARN] Empty data for plot: {title}")
        return

    x = x[:n]
    y_mean = y_mean[:n]
    y_std = y_std[:n]

    plt.figure()
    plt.plot(x, y_mean)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def _pad_to_length(arr, L):
    import numpy as np
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.zeros(L, dtype=float)
    if arr.size >= L:
        return arr[:L]
    pad_val = arr[-1]
    pad = np.full(L - arr.size, pad_val, dtype=arr.dtype)
    return np.concatenate([arr, pad])


def _debug_print_agg_stats(name, agg):
    keys = sorted(list(agg.keys()))
    print(f"[DEBUG] {name} keys: {keys}")
    alive_len = len(agg.get("alive_mean", []))
    avg_energy_len = len(agg.get("avg_energy_mean", []))
    var_energy_len = len(agg.get("var_energy_mean", []))
    print(
        f"[DEBUG] {name} lens -> "
        f"alive_mean: {alive_len}, "
        f"avg_energy_mean: {avg_energy_len}, "
        f"var_energy_mean: {var_energy_len}"
    )


def plot_baselines(agg_random, agg_echp, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)

    _debug_print_agg_stats("Random", agg_random)
    _debug_print_agg_stats("ECHP", agg_echp)

    L = int(max(agg_random["max_len"], agg_echp["max_len"]))
    # Pad curves to same length L
    r_alive = _pad_to_length(agg_random["alive_mean"], L)
    e_alive = _pad_to_length(agg_echp["alive_mean"], L)

    r_avgE = _pad_to_length(agg_random["avg_energy_mean"], L)
    e_avgE = _pad_to_length(agg_echp["avg_energy_mean"], L)

    r_varE = _pad_to_length(agg_random["var_energy_mean"], L)
    e_varE = _pad_to_length(agg_echp["var_energy_mean"], L)
    x = np.arange(len(r_alive))

    # Individual plots (mean only)
    plt.figure()
    plt.plot(x, r_alive)
    plt.title("Alive Nodes vs Rounds (Random)")
    plt.xlabel("Round")
    plt.ylabel("Alive Nodes")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/alive_random.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, e_alive)
    plt.title("Alive Nodes vs Rounds (ECHP-heuristic)")
    plt.xlabel("Round")
    plt.ylabel("Alive Nodes")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/alive_echp.png", dpi=200)
    plt.close()

    # Compare on one plot
    plt.figure()
    plt.plot(x, r_alive, label="Random")
    plt.plot(x, e_alive, label="ECHP")
    plt.title("Alive Nodes vs Rounds (Baselines Comparison)")
    plt.xlabel("Round")
    plt.ylabel("Alive Nodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/alive_compare.png", dpi=200)
    plt.close()

    # Avg energy compare
    plt.figure()
    plt.plot(x, r_avgE, label="Random")
    plt.plot(x, e_avgE, label="ECHP")
    plt.title("Average Residual Energy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Avg Energy (J)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/avg_energy_compare.png", dpi=200)
    plt.close()

    # Variance compare
    plt.figure()
    plt.plot(x, r_varE, label="Random")
    plt.plot(x, e_varE, label="ECHP")
    plt.title("Energy Variance vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Var(E)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/var_energy_compare.png", dpi=200)
    plt.close()

def plot_three_way_compare(agg_random, agg_echp, agg_rl, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)

    _debug_print_agg_stats("Random", agg_random)
    _debug_print_agg_stats("ECHP", agg_echp)
    _debug_print_agg_stats("RL", agg_rl)

    L = int(max(agg_random["max_len"], agg_echp["max_len"], agg_rl["max_len"]))

    r_alive = _pad_to_length(agg_random["alive_mean"], L)
    e_alive = _pad_to_length(agg_echp["alive_mean"], L)
    rl_alive = _pad_to_length(agg_rl["alive_mean"], L)

    r_avgE = _pad_to_length(agg_random["avg_energy_mean"], L)
    e_avgE = _pad_to_length(agg_echp["avg_energy_mean"], L)
    rl_avgE = _pad_to_length(agg_rl["avg_energy_mean"], L)

    r_varE = _pad_to_length(agg_random["var_energy_mean"], L)
    e_varE = _pad_to_length(agg_echp["var_energy_mean"], L)
    rl_varE = _pad_to_length(agg_rl["var_energy_mean"], L)

    x = np.arange(len(r_alive))

    # Alive compare
    plt.figure()
    plt.plot(x, r_alive, label="Random")
    plt.plot(x, e_alive, label="ECHP")
    plt.plot(x, rl_alive, label="RL")
    plt.title("Alive Nodes vs Rounds (Comparison)")
    plt.xlabel("Round")
    plt.ylabel("Alive Nodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/alive_compare.png", dpi=200)
    plt.close()

    # Avg energy compare
    plt.figure()
    plt.plot(x, r_avgE, label="Random")
    plt.plot(x, e_avgE, label="ECHP")
    plt.plot(x, rl_avgE, label="RL")
    plt.title("Average Residual Energy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Avg Energy (J)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/avg_energy_compare.png", dpi=200)
    plt.close()

    # Variance compare
    plt.figure()
    plt.plot(x, r_varE, label="Random")
    plt.plot(x, e_varE, label="ECHP")
    plt.plot(x, rl_varE, label="RL")
    plt.title("Energy Variance vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Var(E)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/var_energy_compare.png", dpi=200)
    plt.close()
