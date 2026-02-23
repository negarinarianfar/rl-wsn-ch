# quick_test.py
from src.energy_model import EnergyParams
from src.wsn_env import EnvConfig, WSNEnv


def main():
    cfg = EnvConfig(
        n_nodes=100,
        area_w=100.0,
        area_h=100.0,
        bs_pos=(50.0, 50.0),
        init_energy=1.0,
        packet_bits=4000,
        topk_candidates=10,
        dead_ratio_terminate=0.8
    )

    # نمونه پارامترها (عددها را بعداً با مقاله‌ات هماهنگ می‌کنیم)
    eparams = EnergyParams(
        E_elec=50e-9,
        eps_fs=10e-12,
        eps_mp=0.0013e-12,
        E_da=5e-9
    )

    env = WSNEnv(cfg, eparams)
    s = env.reset(seed=1)

    # یک سیاست خیلی ساده: همیشه از top-k یکی رو انتخاب کن (اینجا: اولین)
    done = False
    total_r = 0.0
    while not done:
        candidates = env.build_candidates()
        ch = int(candidates[0])  # heuristic dummy
        s, r, done, info = env.step(ch)
        total_r += r
        if info["round"] % 50 == 0:
            print(f"Round={info['round']}, Alive={info['alive']}, Consumed={info['energy_consumed']:.6f}")

    print("Finished.")
    print("Total reward:", total_r)
    print("FND/HND/LND:", info["FND"], info["HND"], info["LND"])

if __name__ == "__main__":
    main()