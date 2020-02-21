from collections import deque

import numpy as np
import torch


def sample_values(values: deque, indices: np.ndarray) -> torch.Tensor:
    return torch.as_tensor([values[i] for i in indices], dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, max_length):
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.new_states = deque(maxlen=max_length)
        self.dones = deque(maxlen=max_length)

    def store(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def sample_batch(self, batch_size: int):
        num_elements = min(self.states.maxlen, len(self.states))
        indices = np.random.randint(low=0, high=num_elements, size=batch_size)
        return {
            "states": sample_values(self.states, indices),
            "actions": sample_values(self.actions, indices),
            "rewards": sample_values(self.rewards, indices),
            "new_states": sample_values(self.new_states, indices),
            "dones": sample_values(self.dones, indices),
        }
