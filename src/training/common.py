import random
from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Union, Optional

import gym
import numpy as np
import torch
import sklearn.preprocessing
from torch.distributions import Categorical, Normal

from src.networks.simple import SimplePolicyDiscrete, SimplePolicyContinuous, SimpleCritic


@dataclass
class RunParams:
    render_frequency: int = 1  # Interval between renders (in number of episodes)
    logging_frequency: int = 1  # Interval between logs (in number of episodes)
    gamma: float = 0.99  # Discount factor
    train_with_batches: bool = False
    batch_size: int = 24
    continuous_actions: bool = True
    should_scale_states: bool = True
    entropy_coeff: float = 1
    entropy_decay: float = 1

    def should_render(self, episode_number: int) -> bool:
        return self.render_frequency > 0 and episode_number % self.render_frequency == 0

    def should_log(self, episode_number: int) -> bool:
        return self.logging_frequency > 0 and episode_number % self.logging_frequency == 0




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
    episode_number: int = 0
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
        self.episode_number += 1

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

            if type(self.actions[0]) == int:
                actions = torch.tensor(self.actions)[indices]
            else:
                actions = torch.cat(self.actions)[indices]
            discounted_rewards = self.discounted_rewards[indices]
            yield states, actions, discounted_rewards


def prepare_state(state: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(state).float().unsqueeze(0)


def select_action_discrete(state, policy: SimplePolicyDiscrete, training_info: TrainingInfo, env: gym.Env):
    # Get distribution
    state = prepare_state(state)
    probs = policy.forward(state)

    # Sample action and remember its log probability
    m = Categorical(probs)
    action = m.sample()

    training_info.log_probs.append(m.log_prob(action))
    training_info.entropies.append(m.entropy())

    return action.item()


def select_action_continuous(state, policy: SimplePolicyContinuous, training_info: TrainingInfo, env: gym.Env):
    # Get distribution
    state = prepare_state(state)
    mu, sigma = policy.forward(state)

    # Sample action and remember its log probability
    n = Normal(mu, sigma)
    action = n.sample()
    action = action.clamp(env.action_space.low[0], env.action_space.high[0])

    # This is not very clean. TODO: clean this up
    training_info.log_probs.append(n.log_prob(action))
    training_info.entropies.append(n.entropy())

    return action


def get_state_value(state, critic: SimpleCritic):
    return critic.forward(prepare_state(state))


def save_model(model: torch.nn.Module, filename: str):
    """ Saves the model in the data directory """
    path = Path().cwd().parent.parent / 'data' / filename
    torch.save(model.state_dict(), path.resolve().as_posix())


def load_model(model_to_fill: torch.nn.Module, filename: str):
    """ Load the model from a weights file inside the data directory"""
    path = Path().cwd().parent.parent / 'data' / filename
    model_to_fill.load_state_dict(torch.load(path.resolve().as_posix()))
    model_to_fill.eval()
    return model_to_fill


def setup_scaler(env: gym.Env) -> sklearn.preprocessing.StandardScaler:
    observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)
    return scaler


def scale_state(scaler: sklearn.preprocessing.StandardScaler, state: np.ndarray) -> np.ndarray:
    return scaler.transform(state.reshape(1, -1))[0]


def run_model(policy: torch.nn.Module, env: gym.Env, continuous_actions: bool = True):
    scaler = setup_scaler(env)
    training_info = TrainingInfo()

    done = False
    while not done:
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(env.spec.max_episode_steps):
            state = scale_state(scaler, state)
            if continuous_actions:
                action = select_action_continuous(state, policy, training_info, env)
            else:
                action = select_action_discrete(state, policy, training_info, env)

            new_state, reward, done, _ = env.step(action)

            env.render()
            state = new_state
            if done:
                break
