# RL-WSN-CH

Reinforcement Learning-based Cluster Head Optimization for Wireless Sensor Networks

## Overview

This project investigates reinforcement learning approaches for adaptive cluster head selection in Wireless Sensor Networks (WSNs).

The proposed framework introduces a multi-role clustering strategy using reinforcement learning to improve energy efficiency, balance energy consumption among nodes, and extend network lifetime.

The system was implemented using a custom simulation environment and evaluated against multiple baseline methods.

---

## Features

- Custom Wireless Sensor Network simulation environment
- Reinforcement Learning-based multi-role cluster head selection
- Energy-aware reward engineering
- Adaptive decision-making framework
- Baseline comparison with heuristic methods
- Evaluation using standard WSN lifetime metrics

---

## Baseline Methods

The proposed RL-MR method is compared against:

- Random Multi-Role Selection
- LEACH-MR
- HEED-MR

---

## Evaluation Metrics

The following metrics are used for evaluation:

- FND (First Node Dies)
- HND (Half Nodes Die)
- Network Lifetime
- Alive Nodes vs Rounds
- Dead Nodes vs Rounds
- Average Residual Energy

---

## Technologies

- Python
- NumPy
- Matplotlib
- Reinforcement Learning
- Q-Learning

---

## Project Structure

```text
src/
│
├── baselines.py
├── config.py
├── energy_model.py
├── evaluate.py
├── metrics.py
├── multi_role_policy.py
├── plot_multirole_results.py
├── rl_agent.py
├── rl_multirole_agent.py
├── run_multirole_baselines.py
├── train_multirole_rl.py
├── wsn_env.py
```

---

## Training

Run RL training:

```bash
python src/train_multirole_rl.py
```

---

## Run Baselines

```bash
python src/run_multirole_baselines.py
```

---

## Generate Figures

```bash
python src/plot_multirole_results.py
```

---

## Sample Results

The framework generates:

- Alive Nodes vs Rounds
- Dead Nodes vs Rounds
- Average Residual Energy Curves
- FND-Lifetime Trade-off Analysis

---

## Research Context

This work extends previous research on energy-efficient clustering in Wireless Sensor Networks by integrating reinforcement learning-based adaptive decision-making.

The project focuses on sequential optimization and energy balancing through intelligent cluster management.

---

## Future Improvements

Potential future extensions include:

- Deep Reinforcement Learning (DQN/PPO)
- Multi-Agent Reinforcement Learning
- Graph Neural Networks for WSN optimization
- Federated Learning-based clustering
- UAV-assisted sensor communication

---

## Author

Negar Arianfar

---

## License

MIT License
