import itertools
import random
from dataclasses import dataclass

import gym
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from networks.simple import SacPolicy, SacValueEstimator
from training.common import RunParams, TrainingInfo, setup_scaler, scale_state, log_on_console, log_on_tensorboard, \
    close_tensorboard, save_model, save_scaler, polyak_average, policy_run, load_model, load_scaler

from training.replay_buffer import ReplayBuffer


@dataclass
class SacParams:
    policy: SacPolicy
    policy_target: SacPolicy
    value_estimator1: SacValueEstimator
    value_estimator2: SacValueEstimator
    value_estimator1_target: SacValueEstimator
    value_estimator2_target: SacValueEstimator

    policy_optimizer: Optimizer
    value_optimizer: Optimizer

    replay_buffer_size: int

    update_frequency: int
    update_start: int
    batch_size: int

    polyak: float  # Polyak coefficient, indicating how much of target is changed

    num_random_action_steps: int  # During the first start_steps steps, we pick random actions

    num_test_episodes: int  # Number of episodes used to evaluate the agent

    alpha: float  # Entropy coefficient in the Bellman equation

    test_frequency: int = 10  # How often we test the agent (in number of episodes)


def compute_value_loss(batch_transitions, sac_params: SacParams, run_params: RunParams):
    states, actions, rewards = batch_transitions["states"], batch_transitions["actions"], batch_transitions["rewards"]
    new_states, dones = batch_transitions["new_states"], batch_transitions["dones"]

    values1 = sac_params.value_estimator1.forward(states, actions)
    values2 = sac_params.value_estimator2.forward(states, actions)

    with torch.no_grad():
        # The actions for the next state come from **current** policy (not from the target policy)
        actions_next_states, log_actions_next_states = sac_params.policy.forward(new_states)

        values_next_state_target1 = sac_params.value_estimator1_target.forward(new_states, actions_next_states)
        values_next_state_target2 = sac_params.value_estimator2_target.forward(new_states, actions_next_states)
        values_next_state_target = torch.min(values_next_state_target1, values_next_state_target2)

        values_expected = rewards + run_params.gamma * (1 - dones) * (values_next_state_target -
                                                                      sac_params.alpha * log_actions_next_states)

    value1_loss = ((values1 - values_expected) ** 2).mean()
    value2_loss = ((values2 - values_expected) ** 2).mean()
    return value1_loss + value2_loss


def compute_policy_loss(batch_transitions, sac_params: SacParams) -> torch.Tensor:
    # Idea: we want to maximize the values of the actions predicted by the policy
    # and to minimize the log likelihood of the actions (policy gradient)
    # We therefore want to do gradient ascent, so we have to negate the loss
    states = batch_transitions["states"]
    actions, log_actions = sac_params.policy.forward(states)
    values1 = sac_params.value_estimator1.forward(states, actions)
    values2 = sac_params.value_estimator2.forward(states, actions)
    values = torch.min(values1, values2)

    return (sac_params.alpha * log_actions - values).mean()


def update_models(batch_transitions, sac_params: SacParams, run_params: RunParams, writer: SummaryWriter,
                  step_number):
    # Update the value function
    sac_params.value_optimizer.zero_grad()
    value_loss = compute_value_loss(batch_transitions, sac_params, run_params)
    value_loss.backward()
    sac_params.value_optimizer.step()

    # To avoid computing gradients we won't need, we can freeze the value-network (src: Spinning Up).
    q_params = itertools.chain(sac_params.value_estimator1.parameters(), sac_params.value_estimator2.parameters())
    for parameter in q_params:
        parameter.requires_grad = False

    # Update the policy network
    sac_params.policy_optimizer.zero_grad()
    policy_loss = compute_policy_loss(batch_transitions, sac_params)
    policy_loss.backward()
    sac_params.policy_optimizer.step()

    # Unfreeze the value estimator parameters, so that further DDPG steps will work
    q_params = itertools.chain(sac_params.value_estimator1.parameters(), sac_params.value_estimator2.parameters())
    for parameter in q_params:
        parameter.requires_grad = True

    # Log things on tensorboard and console if needed
    if run_params.use_tensorboard:
        writer.add_scalar("Loss/Policy", policy_loss.item(), step_number)
        writer.add_scalar("Loss/Value", value_loss.item(), step_number)

    # We now need to update the target networks, given the new weights of the normal networks
    polyak_average(sac_params.policy, sac_params.policy_target, sac_params.polyak)
    polyak_average(sac_params.value_estimator1, sac_params.value_estimator1_target, sac_params.polyak)
    polyak_average(sac_params.value_estimator2, sac_params.value_estimator2_target, sac_params.polyak)


def select_action_sac(state, sac_params: SacParams, deterministic: bool = False,
                      compute_log_prob: bool = False) -> np.ndarray:
    return sac_params.policy.get_actions(torch.tensor(state).float(), deterministic, compute_log_prob)


def test_agent_performance(env: gym.Env, sac_params: SacParams, run_params: RunParams, writer: SummaryWriter,
                           test_episode_number: int, scaler):
    with torch.no_grad():
        episode_rewards, episode_lengths = [], []
        for j in range(sac_params.num_test_episodes):
            state, done, episode_reward, episode_length = env.reset(), False, 0, 0
            while not done:
                state_scaled = scale_state(scaler, state) if run_params.should_scale_states else state
                action = select_action_sac(state_scaled, sac_params, deterministic=True)  # No noise, pure exploitation
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


def sac_train(
        env: gym.Env,
        run_params: RunParams,
        sac_params: SacParams):
    """
    :param env: the OpenAI gym environment
    :param run_params: the general training parameters shared by all training algorithm
    :param sac_params: the DDPG-specific information (networks, optimizers, parameters)
    """
    assert run_params.continuous_actions, "SAC implementation only implemented for continuous action spaces"

    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    # Optimization for speed: don't compute gradients for the target networks, since we will never use them
    for network in [sac_params.policy_target, sac_params.value_estimator1_target, sac_params.value_estimator2_target]:
        for parameter in network.parameters():
            parameter.requires_grad = False

    # Setup tensorboard
    writer = run_params.get_tensorboard_writer(env) if run_params.use_tensorboard else None

    # Setup scaler, training info and replay buffer
    scaler = setup_scaler(env) if run_params.should_scale_states else None
    training_info = TrainingInfo(GAMMA=run_params.gamma)
    replay_buffer = ReplayBuffer(sac_params.replay_buffer_size)

    step_number, test_episode_num = 0, 0
    # TODO: see if this change matters (I think it will, because if the robot has been advancing for a
    # it's possible that it won't crash before the 1000th step, and so the reward will be much better)
    # Thus, it will be incentivized to advance much longer
    max_episode_steps = min(env.spec.max_episode_steps, 10000)

    for episode_number in range(run_params.maximum_episodes):
        state = env.reset()
        episode_length = 0

        # Do a whole episode
        for t in range(max_episode_steps):
            if run_params.should_scale_states:
                state = scale_state(scaler, state)

            # Pick an action, execute and observe the results
            # Note: in the first start_steps steps, we randomly pick actions from
            # the action space (uniformly) to have better exploration.
            if step_number >= sac_params.num_random_action_steps:
                action, log_prob = select_action_sac(state, sac_params, compute_log_prob=True)
            else:
                action = env.action_space.sample()
                log_prob = -1

            # For debugging, log the Q-values
            if run_params.use_tensorboard:
                if random.random() < 0.01:  # Don't log too often to avoid slowing things down
                    s, a = torch.tensor(state).float(), torch.tensor(action).float()
                    value1 = sac_params.value_estimator1.forward(s, a)
                    value2 = sac_params.value_estimator2.forward(s, a)
                    value1_target = sac_params.value_estimator1_target.forward(s, a)
                    value2_target = sac_params.value_estimator2_target.forward(s, a)

                    for action_index in range(a.shape[0]):
                        writer.add_scalar(f"Action/{action_index}", a[action_index], step_number)
                    writer.add_scalar("Q-values/Normal Network 1", value1, step_number)
                    writer.add_scalar("Q-values/Normal Network 2", value2, step_number)
                    writer.add_scalar("Q-values/Target Network 1", value1_target, step_number)
                    writer.add_scalar("Q-values/Target Network 2", value2_target, step_number)
                    writer.add_scalar("Action/Log prob action", log_prob, step_number)

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

            # if step_number >= sac_params.update_start and step_number % sac_params.update_frequency == 0:
            #     for update_step in range(sac_params.update_frequency):
            #         batch_transitions = replay_buffer.sample_batch(sac_params.batch_size)
            #         update_models(batch_transitions, sac_params, run_params, writer, step_number)

            step_number += 1
            episode_length += 1

        # Training at the end of the episode approach taken from
        # https://github.com/createamind/DRL/blob/master/spinup/algos/sac1/sac1_BipedalWalker-v2_200ep.py
        for update_step in range(int(episode_length * 1.5)):
            batch_transitions = replay_buffer.sample_batch(sac_params.batch_size)
            update_models(batch_transitions, sac_params, run_params, writer, step_number)

        if (episode_number + 0) % sac_params.test_frequency == 0:
            test_agent_performance(env, sac_params, run_params, writer, test_episode_num, scaler)
            test_episode_num += 1

        if run_params.should_save_model(episode_number):
            save_model_sac(env, sac_params, scaler)

        training_info.update_running_reward()

        # Add some logging
        log_on_console(env, episode_number, reward, run_params, t, training_info)
        log_on_tensorboard(env, episode_number, reward, run_params, t, training_info, writer)

        # Check if we have solved the environment reliably
        if run_params.stop_at_threshold and env.spec.reward_threshold is not None and training_info.running_reward > env.spec.reward_threshold:
            print(f"Solved! The running reward is {training_info.running_reward:.2f}, which is above the threshold of "
                  f"{env.spec.reward_threshold}. The last episode ran for {t} steps.")
            save_model_sac(env, sac_params, scaler)
            break

        training_info.reset()

    close_tensorboard(run_params, writer)


def save_model_sac(env, sac_params, scaler):
    save_model(sac_params.policy_target, env, "policy_target.data")
    save_model(sac_params.value_estimator1_target, env, "value_estimator1_target.data")
    save_model(sac_params.value_estimator2_target, env, "value_estimator2_target.data")
    if scaler is not None:
        save_scaler(scaler, env, "scaler.data")


def sac_run(env, policy, scaler=None, render=True, run_once=False):
    return policy_run(env, policy, scaler, render, say_deterministic=False, run_once=run_once)


def sac_run_from_disk(env, has_scaler=True, render=True, run_once=False):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ddpg_policy = load_model(
        SacPolicy(state_dim, action_dim, env.action_space.high, env.action_space.low),
        env, "policy_target.data"
    )
    scaler = load_scaler(env, "scaler.data") if has_scaler else None
    sac_run(env, ddpg_policy, scaler=scaler, render=render, run_once=run_once)
