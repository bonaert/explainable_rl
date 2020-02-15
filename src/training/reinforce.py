import itertools
from dataclasses import dataclass, field
from typing import List, Union

import gym
import numpy as np
import sklearn.preprocessing
import torch
from torch.distributions import Categorical, Normal

from src.policy.linear import SimplePolicyDiscrete, SimplePolicyContinuous
from torch.optim.optimizer import Optimizer

RENDER_FREQUENCY = 0  # Interval between renders (in number of episodes)
LOGGING_FREQUENCY = 1  # Interval between logs (in number of episodes)
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 24


@dataclass
class TrainingInfo:
    """ Stores the rewards and log probabilities during an episode """
    log_probs: List[float] = field(default_factory=list)
    states: List[float] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    rewards: Union[List[float], torch.Tensor] = field(default_factory=list)
    discounted_rewards: Union[List[float], torch.Tensor] = field(default_factory=list)
    all_discounted_rewards: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    running_reward: float = None
    episode_reward: float = 0

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.discounted_rewards = []
        self.log_probs.clear()
        self.episode_reward = 0

    def record_step(self, state, action, reward: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
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
            discounted_reward = reward + GAMMA * discounted_reward
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
    training_info.log_probs.append(
        n.log_prob(action) + n.entropy()  # Add entropy to encourage exploration
    )

    return action


def train_policy(optimizer: Optimizer, training_info: TrainingInfo):
    training_info.compute_discounted_rewards()

    # Compute the loss of the policy at each time step
    policy_losses = []
    for log_prob, discounted_reward in zip(training_info.log_probs, training_info.discounted_rewards):
        policy_losses.append(-log_prob * discounted_reward)

    # Optimize the policy
    optimizer.zero_grad()
    total_policy_loss = torch.cat(policy_losses).sum()
    total_policy_loss.backward()
    optimizer.step()

    # Reset the state of the episode
    training_info.reset()


def train_policy_batches(policy: SimplePolicyContinuous, optimizer: Optimizer, training_info: TrainingInfo,
                         episode_number: int):
    training_info.compute_discounted_rewards()

    for (states, actions, discounted_rewards) in training_info.get_batches(BATCH_SIZE):
        train_batch(policy, states, actions, discounted_rewards, optimizer, episode_number)

    training_info.reset()


def train_batch(policy: SimplePolicyContinuous, states, actions, discounted_rewards, optimizer, episode_number):
    optimizer.zero_grad()

    policy_losses = []
    for (state, action, discounted_reward) in zip(states, actions, discounted_rewards):
        state = state.float().unsqueeze(0)
        mu, sigma = policy.forward(state)
        n = Normal(mu, sigma)
        policy_losses.append(-(n.log_prob(action) + 0.99 ** episode_number * n.entropy()) * discounted_reward)

    total_policy_loss = torch.cat(policy_losses).sum()
    total_policy_loss.backward()
    optimizer.step()


def reinforceTraining(
        policy: SimplePolicyDiscrete,
        env: gym.Env,
        optimizer: Optimizer,
        continuous_actions: bool,
        scale_state: bool = False,
        train_with_batches: bool = False):
    """
    :param policy: the policy that picks the action and improves over time
    :param env: the OpenAI gym environment
    :param optimizer: optimizer that improves the policy
    :param continuous_actions: is the action space continuous
    :param scale_state: should the state be scaled before being used by the policy
    :param train_with_batches: trains using mini-batches with entropy (note: only for continuous action spaces problems)
    :return:
    """

    training_info = TrainingInfo()
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    if scale_state:
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)

    for episode_number in itertools.count(1):  # itertools.count(1) is basically range(1, infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(1, 10000):
            if continuous_actions:
                if scale_state:
                    state = scaler.transform(state.reshape(1, -1))

                action = select_action_continuous(state, policy, training_info, env)
            else:
                action = select_action_discrete(state, policy, training_info)

            new_state, reward, done, _ = env.step(action)

            if RENDER_FREQUENCY > 0 and episode_number % RENDER_FREQUENCY == 0:
                env.render()

            training_info.record_step(state, action, reward)  # Store reward and updates the running reward
            state = new_state
            if done:
                break

        training_info.update_running_reward()

        # Add some logging
        if episode_number % LOGGING_FREQUENCY == 0:
            print(f"Episode {episode_number}\t"
                  f"Solved: {t < env.spec.max_episode_steps}\t"
                  f"Average reward: {training_info.episode_reward / t:.2f}\t"
                  f"Episode reward: {training_info.episode_reward:.2f}\t"
                  f"Running Reward: {training_info.running_reward:.2f}\t"
                  f"Number of steps during episode: {t}")

        # Check if we have solved the environment reliably
        if env.spec.reward_threshold is not None and training_info.running_reward > env.spec.reward_threshold:
            print(f"Solved! The running reward is {training_info.running_reward:.2f}, which is above the threshold of "
                  f"{env.spec.reward_threshold}. The last episode ran for {t} steps.")
            break

        if continuous_actions and train_with_batches:
            train_policy_batches(policy, optimizer, training_info, episode_number)
        else:
            train_policy(optimizer, training_info)
