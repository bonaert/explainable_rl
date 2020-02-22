import itertools

import gym
import torch
import numpy as np
import torch.nn.functional as F

from src.networks.simple import SimplePolicyContinuous, SimpleCritic
from torch.optim.optimizer import Optimizer

from src.training.common import select_action_discrete, select_action_continuous, TrainingInfo, get_state_value, \
    prepare_state, setup_observation_scaler, scale_state, RunParams, close_tensorboard
from training.common import log_on_tensorboard, log_on_console


def train_policy_on_episode(optimizer: Optimizer, training_info: TrainingInfo, episode_number: int):
    """ Trains both the actor and the critic using all transitions of the latest episode. The actor's loss is the MSE
     between V(state) and reward + gamma * V(next state), where V indicates the actor's value function.
     The actor / policy is trained by maximizing the log probability * td-error, and an entropy term is
     added to encourage exploration. The entropy is decayed at new each episode by the run_params.entropy_decay
     coefficient.
    """
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
        critic: SimpleCritic,
        optimizer: Optimizer,
        reward: float,
        state: np.ndarray,
        next_state: np.ndarray,
        gamma: float,
        log_prob: float,
        entropy: float,
        episode_number: int,
        run_params: RunParams):
    """ Trains both the actor and the critic using the given transition. The actor's loss is the MSE
     between V(state) and reward + gamma * V(next state), where V indicates the actor's value function.
     The actor / policy is trained by maximizing the log probability * td-error, and an entropy term is
     added to encourage exploration. The entropy is decayed at new each episode by the run_params.entropy_decay
     coefficient.
     """
    # Inspired from https://gym.openai.com/evaluations/eval_gUhDnmlbTKG1qW0jS6HSg/

    state, next_state = prepare_state(state), prepare_state(next_state)

    state_value_target = reward + gamma * critic.forward(next_state)
    state_value_prediction = critic.forward(state)
    td_error = state_value_target - state_value_prediction

    # Update policy
    optimizer.zero_grad()
    loss = -(log_prob + run_params.entropy_coeff * run_params.entropy_decay ** episode_number * entropy) * td_error
    loss += F.mse_loss(state_value_prediction, state_value_target)
    loss.backward()
    optimizer.step()


def actor_critic_train_per_episode(
        policy: SimplePolicyContinuous,
        critic: SimpleCritic,
        env: gym.Env,
        optimizer: Optimizer,
        run_params: RunParams,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None):
    """ Trains the actor critic on the given environment. Training is done at the end of each episode, instead
    of at the end of each step of an episode. This means the agent trains much more frequently.
    Both discrete and continuous actions spaces are supported. Several features can be optionally enabled:
    1) Scaling / normalizing the states / observations
    2) Logging training statistics on Tensorboard
    3) Render the environment periodically (pick render_frequency in the RunParams)
    4) Using a learning rate scheduler
    """
    training_info = TrainingInfo(GAMMA=run_params.gamma)
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    # https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
    # says it's crucial to scale the state
    if run_params.should_scale_states:
        scaler = setup_observation_scaler(env)

    writer = run_params.get_tensorboard_writer(env) if run_params.use_tensorboard else None

    for episode_number in itertools.count():  # itertools.count() is basically range(+infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(env.spec.max_episode_steps):
            if run_params.should_scale_states:
                state = scale_state(scaler, state)

            if run_params.continuous_actions:
                action = select_action_continuous(state, policy, training_info, env)
            else:
                action = select_action_discrete(state, policy, training_info)

            state_value = get_state_value(state, critic)

            new_state, reward, done, _ = env.step(action)

            if run_params.should_render(episode_number):
                env.render()

            training_info.record_step(state, action, reward, state_value)  # Store reward and updates the running reward
            state = new_state
            if done:
                break

        training_info.update_running_reward()

        # Add some logging
        log_on_console(env, episode_number, reward, run_params, t, training_info)
        log_on_tensorboard(env, episode_number, reward, run_params, t, training_info, writer)

        # Check if we have solved the environment reliably
        if env.spec.reward_threshold is not None and training_info.running_reward > env.spec.reward_threshold:
            print(f"Solved! The running reward is {training_info.running_reward:.2f}, which is above the threshold of "
                  f"{env.spec.reward_threshold}. The last episode ran for {t} steps.")
            break

        train_policy_on_episode(optimizer, training_info, episode_number)

        if lr_scheduler:
            lr_scheduler.step(episode_number)

    close_tensorboard(run_params, writer)


def actor_critic_train_per_step(
        policy: SimplePolicyContinuous,
        critic: SimpleCritic,
        env: gym.Env,
        optimizer: Optimizer,
        run_params: RunParams,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None):
    """ Trains the actor critic on the given environment. Training is done at the end of each step, instead of
    at the end of each episode. This means the agent trains less frequently than doing it at each step.
    Both discrete and continuous actions spaces are supported. Several features can be optionally enabled:
    1) Scaling / normalizing the states / observations
    2) Logging training statistics on Tensorboard
    3) Render the environment periodically (pick render_frequency in the RunParams)
    4) Using a learning rate scheduler
    """
    training_info = TrainingInfo(GAMMA=run_params.gamma)
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    if run_params.should_scale_states:
        scaler = setup_observation_scaler(env)

    writer = run_params.get_tensorboard_writer(env) if run_params.use_tensorboard else None

    for episode_number in itertools.count(1):  # itertools.count() is basically range(+infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(env.spec.max_episode_steps):
            scaled_state = scale_state(scaler, state) if run_params.should_scale_states else state
            if run_params.continuous_actions:
                action = select_action_continuous(scaled_state, policy, training_info, env)
            else:
                action = select_action_discrete(scaled_state, policy, training_info)

            state_value = get_state_value(scaled_state, critic)

            new_state, reward, done, _ = env.step(action)
            if run_params.should_render(episode_number):
                env.render()

            scaled_new_state = scale_state(scaler, new_state) if run_params.should_scale_states else new_state
            training_info.record_step(scaled_state, action, reward, state_value)
            train_policy_on_step(critic, optimizer, reward, scaled_state, scaled_new_state, training_info.GAMMA,
                                 training_info.log_probs[-1], training_info.entropies[-1], episode_number, run_params)

            state = new_state
            if done:
                break

        training_info.update_running_reward()

        # Add some logging
        log_on_console(env, episode_number, reward, run_params, t, training_info)
        log_on_tensorboard(env, episode_number, reward, run_params, t, training_info, writer)

        # Check if we have solved the environment reliably
        if env.spec.reward_threshold is not None and training_info.running_reward > env.spec.reward_threshold:
            print(f"Solved! The running reward is {training_info.running_reward:.2f}, which is above the threshold of "
                  f"{env.spec.reward_threshold}. The last episode ran for {t} steps.")
            break

        training_info.reset()

        if lr_scheduler:
            lr_scheduler.step(episode_number)

    close_tensorboard(run_params, writer)
