from src.config import EnvConfig
from src.energy_model import EnergyParams
from src.wsn_env import WSNEnv
from src.multi_role_policy import build_role_pairs

cfg = EnvConfig(
    n_nodes=100,
    area_size=100,
    init_energy=0.5,
    packet_bits=4000,
    max_rounds=2500,
    top_k_candidates=10,
    dead_ratio_terminate=0.8
)

eparams = EnergyParams(
    E_elec=50e-9,
    eps_fs=10e-12,
    eps_mp=0.0013e-12,
    E_da=5e-9
)

env = WSNEnv(cfg, eparams)
state = env.reset(seed=1)

done = False
total_reward = 0

while not done:
    pairs = build_role_pairs(env, top_k=10)

    if not pairs:
        break

    main_ch, deputy_ch = pairs[0]

    state, reward, done, info = env.step_multi_role(main_ch, deputy_ch)
    total_reward += reward

print("Finished")
print("Total reward:", total_reward)
print("FND/HND/LND:", info.get("FND"), info.get("HND"), info.get("LND"))
print(f"Round={info['round']} Alive={info['alive']} Reward={reward:.4f}")