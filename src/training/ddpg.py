from dataclasses import dataclass
from typing import Union, Dict

import gym
import numpy as np
import torch
import sklearn.preprocessing
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from src.networks.simple import DDPGPolicy, DDPGValueEstimator
from src.training.common import RunParams, TrainingInfo, setup_observation_scaler, scale_state, log_on_console, log_on_tensorboard, \
    close_tensorboard, save_model, save_scaler, polyak_average, policy_run, load_model, load_scaler

from src.training.noise import OUNoise, NormalNoise
from src.training.replay_buffer import ReplayBuffer


@dataclass
class DDPGParams:
    """ Networks and parameters that are specific to running the DDPG algorithm. Most parts of the training
    procedure are tweakable, as well as random exploration at the beginning and testing frequency """
    policy: DDPGPolicy
    policy_target: DDPGPolicy
    value_estimator: DDPGValueEstimator
    value_estimator_target: DDPGValueEstimator

    policy_optimizer: Optimizer
    value_optimizer: Optimizer

    replay_buffer_size: int  # Maximum size of the replay buffer
    batch_size: int  # Size of the batches of transitions that are sampled from the replay buffer to train the agent

    update_frequency: int  # How frequently (in terms of steps) should the policy / value estimator be updated
    update_start: int  # After how many steps should the policy / value estimator start to be updated

    polyak: float  # Polyak coefficient, indicating how much of target is changed

    noise_coeff: float  # We add noise_coeff * normal() noise to the actions
    noise_source: Union[OUNoise, NormalNoise]  # Source of noise that is added to the actions

    num_random_action_steps: int  # During the first start_steps steps, we pick random actions

    num_test_episodes: int  # Number of episodes used to evaluate the agent

    test_frequency: int = 10  # How often we test the agent (in number of episodes)


def compute_policy_loss(batch_transitions: Dict[str, torch.Tensor], ddpg_params: DDPGParams) -> torch.Tensor:
    """ Since the goal is to maximize the values of the actions predicted by the policy, the value loss is defined
    as -(mean of the Q(state, action predicted by policy)). Since we want to do gradient ascent but Pytorch does
    gradient descent, we need to add that minus sign. """
    states = batch_transitions["states"]
    actions = ddpg_params.policy.forward(states)
    values = ddpg_params.value_estimator.forward(states, actions)
    return -values.mean()


def compute_value_loss(batch_transitions: Dict[str, torch.Tensor], ddpg_params: DDPGParams, run_params: RunParams):
    """ Computes the value loss for both value estimators and return the summed losses. For each transition
    in the mini-batch, we compute which action a' the current policy would take at the next state s'. This information
    is used in the update rule. In DDPG, the error for a specific transition is defined as:
        Q(s, a) - (reward + (1 - done) * gamma * Q(s', a'))
    In this implementation, the loss for a specific network is defined as the smooth L1 loss over all transitions
    in the mini-batch (instead of the more conventional MSE)."""
    # TODO: justify why L1 instead of MSE, and see if it actually matters or not
    states, actions, rewards = batch_transitions["states"], batch_transitions["actions"], batch_transitions["rewards"]
    new_states, dones = batch_transitions["new_states"], batch_transitions["dones"]

    values = ddpg_params.value_estimator.forward(states, actions)

    with torch.no_grad():
        actions_target_next_states = ddpg_params.policy_target.forward(new_states)
        values_target = ddpg_params.value_estimator_target.forward(new_states, actions_target_next_states)
        values_expected = rewards + run_params.gamma * (1 - dones) * values_target

    return F.smooth_l1_loss(values, values_expected)


def update_models(batch_transitions: Dict[str, torch.Tensor], ddpg_params: DDPGParams, run_params: RunParams,
                  writer: SummaryWriter, step_number: int):
    """ Updates both the policy and the 2 value networks, and then polyak-updates the corresponding target
    networks. Polyak updating means slightly change the weights of the target network to take into account the
    new weights of the main networks. See polyak_average() for details. """
    # Update the value function
    ddpg_params.value_optimizer.zero_grad()
    value_loss = compute_value_loss(batch_transitions, ddpg_params, run_params)
    value_loss.backward()
    ddpg_params.value_optimizer.step()

    # To avoid computing gradients we won't need, we can freeze the value-network (src: Spinning Up).
    for parameter in ddpg_params.value_estimator.parameters():
        parameter.requires_grad = False

    # Update the policy network
    ddpg_params.policy_optimizer.zero_grad()
    policy_loss = compute_policy_loss(batch_transitions, ddpg_params)
    policy_loss.backward()
    ddpg_params.policy_optimizer.step()

    # Unfreeze the value estimator parameters, so that further DDPG steps will work
    for parameter in ddpg_params.value_estimator.parameters():
        parameter.requires_grad = True

    # Log things on tensorboard and console if needed
    if run_params.use_tensorboard:
        writer.add_scalar("Loss/Policy", policy_loss.item(), step_number)
        writer.add_scalar("Loss/Value", value_loss.item(), step_number)

    # We now need to update the target networks, given the new weights of the normal networks
    polyak_average(ddpg_params.policy, ddpg_params.policy_target, ddpg_params.polyak)
    polyak_average(ddpg_params.value_estimator, ddpg_params.value_estimator_target, ddpg_params.polyak)


def select_action_ddpg(state, ddpg_params: DDPGParams, env: gym.Env, noise_coeff: float) -> np.ndarray:
    """ Select an action using the DDPG policy and then noise is added to it. The noisy action will always be
    contained within the action space. """
    action = ddpg_params.policy.get_actions(torch.tensor(state).float())
    action += noise_coeff * ddpg_params.noise_source.sample()  # Gaussian noise
    action = np.clip(action, env.action_space.low, env.action_space.high)  # Clamp the action inside the action space
    return action


def test_agent_performance(env: gym.Env, ddpg_params: DDPGParams, run_params: RunParams, writer: SummaryWriter,
                           test_episode_number: int, scaler: sklearn.preprocessing.StandardScaler):
    """ Tests the agent's performance by running the policy during a certain amount of episodes. The
        average episode reward and episode length are logged on the console and optionally on Tensorboard"""
    episode_rewards, episode_lengths = [], []
    for j in range(ddpg_params.num_test_episodes):
        state, done, episode_reward, episode_length = env.reset(), False, 0, 0
        while not done:
            state_scaled = scale_state(scaler, state) if run_params.should_scale_states else state
            action = select_action_ddpg(state_scaled, ddpg_params, env, 0)  # No noise, pure exploitation
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    print(f"\tAverage test performance: {np.mean(episode_rewards):.3f}"
          f"\tAverage Episode Steps {np.mean(episode_lengths):.3f}")

    if run_params.use_tensorboard:
        writer.add_scalar("Test Performance/Average Performance", np.mean(episode_rewards), test_episode_number)
        writer.add_scalar("Test Performance/Average Episode Steps", np.mean(episode_lengths), test_episode_number)


def ddpg_train(env: gym.Env, run_params: RunParams, ddpg_params: DDPGParams):
    """
    :param env: the OpenAI gym environment
    :param run_params: the general training parameters shared by all training algorithm
    :param ddpg_params: the DDPG-specific information (networks, optimizers, parameters)
    """
    assert run_params.continuous_actions, "DDPG implementation only implemented for continuous action spaces"

    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    # Optimization for speed: don't compute gradients for the target networks, since we will never use them
    for network in [ddpg_params.policy_target, ddpg_params.value_estimator_target]:
        for parameter in network.parameters():
            parameter.requires_grad = False

    # Setup tensorboard
    writer = run_params.get_tensorboard_writer(env) if run_params.use_tensorboard else None

    # Setup scaler, training info and replay buffer
    scaler = setup_observation_scaler(env) if run_params.should_scale_states else None
    training_info = TrainingInfo(GAMMA=run_params.gamma)
    replay_buffer = ReplayBuffer(ddpg_params.replay_buffer_size)

    step_number, test_episode_num = 0, 0
    max_episode_steps = env.spec.max_episode_steps
    value_time_step = 0

    for episode_number in range(run_params.maximum_episodes):
        state = env.reset()

        # Do a whole episode
        for t in range(max_episode_steps):
            if run_params.should_scale_states:
                state = scale_state(scaler, state)

            # Pick an action, execute and observe the results
            # Note: in the first start_steps steps, we randomly pick actions from
            # the action space (uniformly) to have better exploration.
            if step_number >= ddpg_params.num_random_action_steps:
                action = select_action_ddpg(state, ddpg_params, env, ddpg_params.noise_coeff * 0.995 ** episode_number)
            else:
                action = env.action_space.sample()

            # For debugging, log the Q-values
            if run_params.use_tensorboard:
                s, a = torch.tensor(state).float(), torch.tensor(action).float()
                value = ddpg_params.value_estimator.forward(s, a)
                value_target = ddpg_params.value_estimator_target.forward(s, a)

                for action_index in range(a.shape[0]):
                    writer.add_scalar(f"Action/{action_index}", a[action_index], value_time_step)
                writer.add_scalar("Q-values/Normal Network", value, value_time_step)
                writer.add_scalar("Q-values/Target Network", value_target, value_time_step)

                value_time_step += 1

            new_state, reward, done, _ = env.step(action)

            # Render the environment if wanted
            if run_params.should_render(episode_number):
                env.render()

            # Store reward and updates the running reward
            training_info.record_step(state, action, reward)

            # Add the transition to the replay buffer
            new_state_scaled = scale_state(scaler, new_state) if run_params.should_scale_states else new_state
            replay_buffer.store(state, action, reward, new_state_scaled, done and t < max_episode_steps - 1)

            state = new_state
            if done:
                break

            if step_number >= ddpg_params.update_start and step_number % ddpg_params.update_frequency == 0:
                for update_step in range(ddpg_params.update_frequency):
                    batch_transitions = replay_buffer.sample_batch(ddpg_params.batch_size)
                    update_models(batch_transitions, ddpg_params, run_params, writer, step_number)

            step_number += 1

        if episode_number % ddpg_params.test_frequency == 0:
            test_agent_performance(env, ddpg_params, run_params, writer, test_episode_num, scaler)
            test_episode_num += 1

        if run_params.should_save_model(episode_number):
            save_model_ddpg(ddpg_params, env, scaler)

        training_info.update_running_reward()

        # Add some logging
        log_on_console(env, episode_number, reward, run_params, t, training_info)
        log_on_tensorboard(env, episode_number, reward, run_params, t, training_info, writer)

        # Check if we have solved the environment reliably
        if run_params.stop_at_threshold and env.spec.reward_threshold is not None and training_info.running_reward > env.spec.reward_threshold:
            print(f"Solved! The running reward is {training_info.running_reward:.2f}, which is above the threshold of "
                  f"{env.spec.reward_threshold}. The last episode ran for {t} steps.")
            break

        training_info.reset()
        ddpg_params.noise_source.reset()

    close_tensorboard(run_params, writer)


def save_model_ddpg(ddpg_params, env, scaler):
    """ Saves the DDPG model (and optionally the scaler, if it's not None) to disk in the data directory"""
    save_model(ddpg_params.policy_target, env, "policy_target.data")
    save_model(ddpg_params.value_estimator_target, env, "value_estimator_target.data")
    if scaler is not None:
        save_scaler(scaler, env, "scaler.data")


def ddpg_run(env, policy, scaler=None, render=True, run_once=False):
    """ Run the given DDPG-trained policy (using optionally a observation / state scaler) on the environment
    indefinitely (by default) or over a single episode (if desired). By default the environment is rendered,
    but this can be disabled. """
    return policy_run(env, policy, scaler, render, run_once)


def get_policy_and_scaler(env, has_scaler):
    """ Loads a SAC-trained policy (and optionally a observation / state scaler)"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ddpg_policy = load_model(
        DDPGPolicy(state_dim, action_dim, env.action_space.high, env.action_space.low),
        env, "policy_target.data"
    )
    scaler = load_scaler(env, "scaler.data") if has_scaler else None
    return ddpg_policy, scaler


def ddpg_run_from_disk(env, has_scaler=True, render=True):
    """ Loads a DDPG-trained policy (and optionally a observation / state scaler) and then runs them
    on the environment indefinitely (by default) or over a single episode (if desired).
    By default the environment is rendered, but this can be disabled. """
    ddpg_policy, scaler = get_policy_and_scaler(env, has_scaler)
    ddpg_run(env, ddpg_policy, scaler=scaler, render=render)
