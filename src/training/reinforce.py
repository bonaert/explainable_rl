import itertools
from dataclasses import dataclass, field
from typing import List

import gym
import torch
from torch.distributions import Categorical

from src.policy.linear import SimplePolicy
from torch.optim.optimizer import Optimizer

SHOULD_RENDER = False
LOGGING_FREQUENCY = 1  # Interval between logs (in number of episodes)
GAMMA = 0.99  # Discount factor


@dataclass
class TrainingInfo:
    """ Stores the rewards and log probabilities during an episode """
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    running_reward: float = None
    episode_reward: float = 0

    def reset(self):
        self.log_probs.clear()
        self.rewards.clear()
        self.episode_reward = 0

    def add_reward(self, reward: float):
        self.rewards.append(reward)
        self.episode_reward += reward

    def update_running_reward(self):
        if self.running_reward is None:  # First episode ever
            self.running_reward = self.episode_reward
        else:
            self.running_reward = 0.05 * self.episode_reward + 0.95 * self.running_reward


def select_action(state, policy: SimplePolicy, training_info: TrainingInfo):
    # Get distribution
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy.forward(state)

    # Sample action and remember its log probability
    m = Categorical(probs)
    action = m.sample()
    training_info.log_probs.append(m.log_prob(action))

    return action.item()


def train_policy(optimizer: Optimizer, training_info: TrainingInfo):
    # Compute discounted rewards at each step
    discounted_rewards = []
    discounted_reward = 0
    for reward in training_info.rewards[::-1]:
        discounted_reward = reward + GAMMA * discounted_reward
        discounted_rewards.insert(0, discounted_reward)

    # Normalize the discounted rewards
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    # Compute the loss of the policy at each time step
    policy_losses = []
    for log_prob, discounted_reward in zip(training_info.log_probs, discounted_rewards):
        policy_losses.append(-log_prob * discounted_reward)

    # Optimize the policy
    optimizer.zero_grad()
    total_policy_loss = torch.cat(policy_losses).sum()
    total_policy_loss.backward()
    optimizer.step()

    # Reset the state of the episode
    training_info.reset()

    pass


def reinforceTraining(policy: SimplePolicy, env: gym.Env, optimizer: Optimizer):
    training_info = TrainingInfo()
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    for episode_number in itertools.count(1):  # itertools.count(1) is basically range(infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(1, 10000):
            action = select_action(state, policy, training_info)
            state, reward, done, _ = env.step(action)

            if SHOULD_RENDER:
                env.render()

            training_info.add_reward(reward)  # Store reward and updates the running reward

            if done:
                break

        training_info.update_running_reward()
        train_policy(optimizer, training_info)

        # Add some logging
        if episode_number % LOGGING_FREQUENCY == 0:
            print(f"Episode {episode_number}\t"
                  f"Last reward: {training_info.episode_reward:.2f}\t"
                  f"Average Reward: {training_info.running_reward:.2f}\t"
                  f"Number of steps during episode: {t}")

        # Check if we have solved the environment
        if training_info.running_reward > env.spec.reward_threshold:
            print(f"Solved! The running reward is {training_info.running_reward:.2f}, which is above the threshold of "
                  f"{env.spec.reward_threshold}. The last episode ran for {t} steps.")
            break


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.seed(50)
    torch.manual_seed(50)
    simple_policy = SimplePolicy(input_size=4, output_size=2)
    optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-2)

    reinforceTraining(simple_policy, env, optimizer)
