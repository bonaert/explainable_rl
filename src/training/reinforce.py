import itertools

import gym
import numpy as np
import sklearn.preprocessing
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer

from src.networks.simple import SimplePolicyDiscrete, SimplePolicyContinuous
from src.training.common import select_action_continuous, select_action_discrete, TrainingInfo

RENDER_FREQUENCY = 0  # Interval between renders (in number of episodes)
LOGGING_FREQUENCY = 1  # Interval between logs (in number of episodes)
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 24


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

    training_info = TrainingInfo(GAMMA=GAMMA)
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    if scale_state:
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)

    for episode_number in itertools.count():  # itertools.count() is basically range(+infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(env.spec.max_episode_steps):
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
                  f"Solved: {t < env.spec.max_episode_steps - 1}\t"
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
