from dataclasses import field, dataclass
from typing import List, Union, Optional

import gym
import torch
from torch.distributions import Categorical, Normal

from src.networks.simple import SimplePolicyDiscrete, SimplePolicyContinuous, SimpleCriticContinuous


@dataclass
class TrainingInfo:
    """ Stores the rewards and log probabilities during an episode """
    log_probs: List[float] = field(default_factory=list)
    states: List[float] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    rewards: Union[List[float], torch.Tensor] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)
    state_values: List[torch.Tensor] = field(default_factory=list)
    discounted_rewards: Union[List[float], torch.Tensor] = field(default_factory=list)
    all_discounted_rewards: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    running_reward: float = None
    episode_reward: float = 0
    GAMMA: float = 0.99

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.discounted_rewards = []
        self.log_probs.clear()
        self.state_values.clear()
        self.episode_reward = 0

    def record_step(self,
                    state,
                    action,
                    reward: float,
                    state_value: Optional[torch.Tensor] = None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.episode_reward += reward

    def update_running_reward(self):
        if self.running_reward is None:  # First episode ever
            self.running_reward = self.episode_reward
        else:
            self.running_reward = 0.05 * self.episode_reward + 0.95 * self.running_reward

    def compute_discounted_rewards(self):
        # Compute discounted rewards at each step
        self.discounted_rewards = []
        discounted_reward = 0
        for reward in self.rewards[::-1]:
            discounted_reward = reward + self.GAMMA * discounted_reward
            self.discounted_rewards.insert(0, discounted_reward)

        # Normalize the discounted rewards
        self.discounted_rewards = torch.tensor(self.discounted_rewards)
        self.all_discounted_rewards = torch.cat([self.all_discounted_rewards, self.discounted_rewards])
        self.discounted_rewards = (self.discounted_rewards - self.all_discounted_rewards.mean()) / \
                                  (self.all_discounted_rewards.std() + 1e-9)

    def get_batches(self, batch_size: int):
        permutation = torch.randperm(self.discounted_rewards.shape[0])
        for i in range(0, self.discounted_rewards.shape[0], batch_size):
            indices = permutation[i: i + batch_size]
            states = torch.tensor(self.states)[indices]
            actions = torch.cat(self.actions)[indices]
            discounted_rewards = self.discounted_rewards[indices]
            yield states, actions, discounted_rewards


def select_action_discrete(state, policy: SimplePolicyDiscrete, training_info: TrainingInfo):
    # Get distribution
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy.forward(state)

    # Sample action and remember its log probability
    m = Categorical(probs)
    action = m.sample()
    training_info.log_probs.append(
        m.log_prob(action)
    )

    return action.item()


def select_action_continuous(state, policy: SimplePolicyContinuous, training_info: TrainingInfo, env: gym.Env):
    # Get distribution
    state = torch.from_numpy(state).float().unsqueeze(0)
    mu, sigma = policy.forward(state)

    # Sample action and remember its log probability
    n = Normal(mu, sigma)
    action = n.sample()
    action = action.clamp(env.action_space.low[0], env.action_space.high[0])

    # This is not very clean. TODO: clean this up
    training_info.log_probs.append(n.log_prob(action))
    training_info.entropies.append(n.entropy())

    return action


def get_state_value(state, critic: SimpleCriticContinuous):
    state = torch.from_numpy(state).float().unsqueeze(0)
    return critic.forward(state)
