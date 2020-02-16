import itertools

import gym
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.preprocessing

from src.networks.simple import SimplePolicyContinuous, SimpleCriticContinuous
from torch.optim.optimizer import Optimizer

from src.training.common import select_action_discrete, select_action_continuous, TrainingInfo, get_state_value, \
    prepare_state, setup_scaler, scale_state

RENDER_FREQUENCY = 10  # Interval between renders (in number of episodes)
LOGGING_FREQUENCY = 1  # Interval between logs (in number of episodes)
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 24


def train_policy_on_episode(optimizer: Optimizer, training_info: TrainingInfo, episode_number: int):
    training_info.compute_discounted_rewards()

    # Compute the loss of the policy and the critic at each time step
    policy_losses = []  # Policy errors
    value_losses = []  # Critic errors
    for log_prob, discounted_reward, state_value, entropy in zip(
            training_info.log_probs,
            training_info.discounted_rewards,
            training_info.state_values,
            training_info.entropies):
        advantage = discounted_reward - state_value.item()
        policy_losses.append(-(log_prob + 0.99 ** episode_number * entropy) * advantage)
        value_losses.append(F.smooth_l1_loss(state_value.squeeze(0), torch.tensor([discounted_reward])))

    # Optimize the policy
    optimizer.zero_grad()
    total_policy_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    total_policy_loss.backward()
    optimizer.step()

    # Reset the state of the episode
    training_info.reset()


def train_policy_on_step(
        critic: SimpleCriticContinuous,
        optimizer: Optimizer,
        reward,
        state,
        next_state,
        gamma,
        log_prob,
        entropy):
    # Inspired from https://gym.openai.com/evaluations/eval_gUhDnmlbTKG1qW0jS6HSg/

    state, next_state = prepare_state(state), prepare_state(next_state)

    state_value_target = reward + gamma * critic.forward(next_state)
    state_value_prediction = critic.forward(state)
    td_error = state_value_target - state_value_prediction

    # Update policy
    # loss = -(log_prob + 0.99 ** episode_number * entropy) * td_error
    optimizer.zero_grad()
    loss = -(log_prob + 0.1 * entropy) * td_error + F.mse_loss(state_value_prediction, state_value_target)
    loss.backward()
    optimizer.step()


def actor_critic_train_per_episode(
        policy: SimplePolicyContinuous,
        critic: SimpleCriticContinuous,
        env: gym.Env,
        optimizer: Optimizer,
        continuous_actions: bool,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None):
    """
    :param policy: the policy that picks the action and improves over time
    :param critic: the critic that is used to evaluate each state
    :param env: the OpenAI gym environment
    :param optimizer: optimizer that improves the policy
    :param continuous_actions: is the action space continuous
    :param scheduler:
    """
    training_info = TrainingInfo(GAMMA=GAMMA)
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    # https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
    # says it's crucial to scale the state
    scaler = setup_scaler(env)

    for episode_number in itertools.count():  # itertools.count() is basically range(+infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(env.spec.max_episode_steps):
            state = scale_state(scaler, state)
            if continuous_actions:
                action = select_action_continuous(state, policy, training_info, env)
            else:
                action = select_action_discrete(state, policy, training_info)

            state_value = get_state_value(state, critic)

            new_state, reward, done, _ = env.step(action)

            if RENDER_FREQUENCY > 0 and episode_number % RENDER_FREQUENCY == 0:
                env.render()

            training_info.record_step(state, action, reward, state_value)  # Store reward and updates the running reward
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

        train_policy_on_episode(optimizer, training_info, episode_number)

        if scheduler:
            scheduler.step(episode_number)


def actor_critic_train_per_step(
        policy: SimplePolicyContinuous,
        critic: SimpleCriticContinuous,
        env: gym.Env,
        optimizer: Optimizer,
        continuous_actions: bool,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None):
    """
    :param policy: the policy that picks the action and improves over time
    :param critic: the critic that is used to evaluate each state
    :param env: the OpenAI gym environment
    :param optimizer: optimizer that improves the policy
    :param continuous_actions: is the action space continuous
    :param scheduler:


    https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
    says it's crucial to scale the state
    """
    training_info = TrainingInfo(GAMMA=GAMMA)
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    scaler = setup_scaler(env)

    for episode_number in itertools.count(1):  # itertools.count() is basically range(+infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(env.spec.max_episode_steps):
            scaled_state = scale_state(scaler, state)
            if continuous_actions:
                action = select_action_continuous(scaled_state, policy, training_info, env)
            else:
                action = select_action_discrete(scaled_state, policy, training_info)

            state_value = get_state_value(scaled_state, critic)

            new_state, reward, done, _ = env.step(action)

            if RENDER_FREQUENCY > 0 and episode_number % RENDER_FREQUENCY == 0:
                env.render()

            scaled_new_state = scale_state(scaler, new_state)
            training_info.record_step(scaled_state, action, reward, state_value)
            train_policy_on_step(critic, optimizer, reward, scaled_state, scaled_new_state, training_info.GAMMA,
                                 training_info.log_probs[-1], training_info.entropies[-1])

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

        training_info.reset()

        if scheduler:
            scheduler.step(episode_number)
