import pickle
import numpy as np
from src.multi_role_policy import build_role_triples


class MultiRoleQAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def discretize_state(self, env):
        alive_ratio = int(10 * np.sum(env.energy > 0) / env.cfg.n_nodes)

        mean_energy = int(10 * np.mean(env.energy))

        std_energy = int(10 * np.std(env.energy))

        round_bin = int(env.round_idx / 100)

        return (
            alive_ratio,
            mean_energy,
            std_energy,
            round_bin
        )

    def select_action(self, env, training=True):
        actions = build_role_triples(env, top_k=env.cfg.top_k_candidates)

        if not actions:
            return None, None, None

        state = self.discretize_state(env)

        if training and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(actions))
        else:
            q_values = [self.q.get((state, i), 0.0) for i in range(len(actions))]
            action_idx = int(np.argmax(q_values))

        main_ch, deputy_ch, relay_ch = actions[action_idx]

        return state, action_idx, (main_ch, deputy_ch, relay_ch)

    def update(self, state, action_idx, reward, next_state, next_num_actions, done):
        old_q = self.q.get((state, action_idx), 0.0)

        if done or next_num_actions == 0:
            target = reward
        else:
            next_q = max(self.q.get((next_state, i), 0.0) for i in range(next_num_actions))
            target = reward + self.gamma * next_q

        self.q[(state, action_idx)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q = pickle.load(f)