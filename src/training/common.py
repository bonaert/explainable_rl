import random
from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Union, Optional

import gym
import numpy as np
import torch
import sklearn.preprocessing
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
import joblib

from datetime import datetime

from src.networks.simple import SimplePolicyDiscrete, SimplePolicyContinuous, SimpleCritic


def tensor_clamp(x: torch.tensor, minimum: np.ndarray, maxixum: np.ndarray):
    return torch.max(torch.min(x, torch.as_tensor(maxixum)), torch.as_tensor(minimum))


def get_env_name(env: gym.Env):
    return env.unwrapped.spec.id


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

    use_tensorboard: bool = False
    tensorboard_log_dir: str = None

    env_can_be_solved: bool = True

    save_model_frequency: int = 0  # How frequently (in episodes) should we save the model (0 = never save)

    stop_at_threshold: bool = True
    maximum_episodes: int = 10000000000000000

    def should_render(self, episode_number: int) -> bool:
        return self.render_frequency > 0 and episode_number % self.render_frequency == 0

    def should_log(self, episode_number: int) -> bool:
        return self.logging_frequency > 0 and episode_number % self.logging_frequency == 0

    def should_save_model(self, episode_number: int):
        return self.save_model_frequency > 0 and episode_number % self.save_model_frequency == 0

    def get_tensorboard_writer(self, env: gym.Env) -> SummaryWriter:
        if self.tensorboard_log_dir is not None:
            return SummaryWriter(f"runs/{get_env_name(env)}/{self.tensorboard_log_dir}")
        else:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            return SummaryWriter(f"runs/{get_env_name(env)}/{current_time}")


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
    action = tensor_clamp(action, env.action_space.low, env.action_space.high)

    # This is not very clean. TODO: clean this up
    training_info.log_probs.append(n.log_prob(action).sum())
    training_info.entropies.append(n.entropy())

    return action


def get_state_value(state, critic: SimpleCritic):
    return critic.forward(prepare_state(state))


def get_path(env, filename, start_path):
    path = start_path.parent.parent / 'data' / get_env_name(env) / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    full_path = path.resolve().as_posix()
    return full_path


def save_model(model: torch.nn.Module, env: gym.Env, filename: str):
    torch.save(model.state_dict(), get_path(env, filename, Path.cwd()))


def save_scaler(scaler, env: gym.Env, filename: str):
    joblib.dump(scaler, get_path(env, filename, Path.cwd()))


def load_model(model_to_fill: torch.nn.Module, env: gym.Env, filename: str):
    full_path = get_path(env, filename, Path.cwd().parent)
    model_to_fill.load_state_dict(torch.load(full_path))
    model_to_fill.eval()
    return model_to_fill


def load_scaler(env: gym.Env, filename: str):
    full_path = get_path(env, filename, Path.cwd().parent)
    return joblib.load(full_path)


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


def log_on_tensorboard(env, episode_number, reward, run_params, t, training_info, writer):
    if run_params.use_tensorboard:
        if run_params.env_can_be_solved:
            writer.add_scalar("Data/Solved", t < env.spec.max_episode_steps - 1, episode_number)

        writer.add_scalar("Data/Average reward", float(training_info.episode_reward) / t, episode_number)
        writer.add_scalar("Data/Episode reward", float(training_info.episode_reward), episode_number)
        writer.add_scalar("Data/Running reward", float(training_info.running_reward), episode_number)
        writer.add_scalar("Data/Last reward", float(reward), episode_number)
        writer.add_scalar("Data/Number of steps per episode", t, episode_number)
        writer.flush()


def log_on_console(env, episode_number, reward, run_params: RunParams, t, training_info):
    if run_params.should_log(episode_number):
        solvedMessage = f"Solved: {t < env.spec.max_episode_steps - 1}\t" if run_params.env_can_be_solved else ""
        print(f"Episode {episode_number}\t"
              f"{solvedMessage}"
              f"Avg. reward: {float(training_info.episode_reward) / t:.2f}\t"
              f"Episode reward: {float(training_info.episode_reward):.2f}\t"
              f"Running Reward: {float(training_info.running_reward):.2f}\t"
              f"Last Reward: {float(reward):.2f}\t"
              f"# Steps in episode: {t}")


def close_tensorboard(run_params, writer):
    if run_params.use_tensorboard:
        writer.flush()
        writer.close()
