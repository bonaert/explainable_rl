from collections import deque
from dataclasses import dataclass
from typing import Union

import gym
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from networks.simple import DDPGPolicy, DDPGValueEstimator
from training.common import RunParams, TrainingInfo, setup_scaler, scale_state, log_on_console, log_on_tensorboard, \
    close_tensorboard, save_model, save_scaler

from training.noise import OUNoise, NormalNoise


@dataclass
class DDPGParams:
    policy: DDPGPolicy
    policy_target: DDPGPolicy
    value_estimator: DDPGValueEstimator
    value_estimator_target: DDPGValueEstimator

    policy_optimizer: Optimizer
    value_optimizer: Optimizer

    replay_buffer_size: int

    update_frequency: int
    update_start: int
    batch_size: int

    polyak: float  # Polyak coefficient, indicating how much of target is changed

    noise_coeff: float  # We add noise_coeff * normal() noise to the actions
    noise_source: Union[OUNoise, NormalNoise]  # Source of noise

    num_random_action_steps: int  # During the first start_steps steps, we pick random actions

    num_test_episodes: int  # Number of episodes used to evaluate the agent

    test_frequency: int = 10  # How often we test the agent (in number of episodes)


def sample_values(values: deque, indices: np.ndarray) -> torch.Tensor:
    return torch.as_tensor([values[i] for i in indices], dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, max_length):
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.new_states = deque(maxlen=max_length)
        self.dones = deque(maxlen=max_length)

    def store(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)

    def sample_batch(self, batch_size: int):
        num_elements = min(self.states.maxlen, len(self.states))
        indices = np.random.randint(low=0, high=num_elements, size=batch_size)
        return {
            "states": sample_values(self.states, indices),
            "actions": sample_values(self.actions, indices),
            "rewards": sample_values(self.rewards, indices),
            "new_states": sample_values(self.new_states, indices),
            "dones": sample_values(self.dones, indices),
        }


def compute_policy_loss(batch_transitions, ddpg_params: DDPGParams) -> torch.Tensor:
    # Idea: we want to maximize the values of the actions predicted by the policy
    # We therefore want to do gradient ascent, so we have to negate the loss
    states = batch_transitions["states"]
    actions = ddpg_params.policy.forward(states)
    values = ddpg_params.value_estimator.forward(states, actions)
    return -values.mean()


def compute_value_loss(batch_transitions, ddpg_params: DDPGParams, run_params: RunParams):
    # Idea: we compute
    # 1) the current values predicted by the value estimator
    # 2) the values we'd should have, using the target value estimator and the target policy
    # The loss is given by the MSE between (1) and (2)
    states, actions, rewards = batch_transitions["states"], batch_transitions["actions"], batch_transitions["rewards"]
    new_states, dones = batch_transitions["new_states"], batch_transitions["dones"]

    values = ddpg_params.value_estimator.forward(states, actions)

    with torch.no_grad():
        actions_target_next_states = ddpg_params.policy_target.forward(new_states)
        values_target = ddpg_params.value_estimator_target.forward(new_states, actions_target_next_states)
        values_expected = rewards + run_params.gamma * (1 - dones) * values_target

    return F.smooth_l1_loss(values, values_expected)
    #return ((values - values_expected) ** 2).mean()


def update_models(batch_transitions, ddpg_params: DDPGParams, run_params: RunParams, writer: SummaryWriter,
                  step_number):
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
    with torch.no_grad():
        for (normal, target) in [(ddpg_params.policy, ddpg_params.policy_target),
                                 (ddpg_params.value_estimator, ddpg_params.value_estimator_target)]:
            for param, param_target in zip(normal.parameters(), target.parameters()):
                # We use the inplace operators to avoid creating new tensor
                param_target.data.mul_(ddpg_params.polyak)
                param_target.data.add_((1 - ddpg_params.polyak) * param.data)


def select_action_ddpg(state, ddpg_params: DDPGParams, env: gym.Env, noise_coeff: float) -> np.ndarray:
    action = ddpg_params.policy.get_actions(torch.tensor(state).float())
    action += noise_coeff * ddpg_params.noise_source.sample()  # Gaussian noise
    action = np.clip(action, env.action_space.low, env.action_space.high)  # Clamp the action inside the action space
    return action


def test_agent_performance(env: gym.Env, ddpg_params: DDPGParams, run_params: RunParams, writer: SummaryWriter,
                           test_episode_number: int, scaler):
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


def ddpg_train(
        env: gym.Env,
        run_params: RunParams,
        ddpg_params: DDPGParams,
        stop_at_threshold: bool = True):
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
    scaler = setup_scaler(env) if run_params.should_scale_states else None
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
                action = select_action_ddpg(state, ddpg_params, env, ddpg_params.noise_coeff * 0.99 ** episode_number)
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
            save_model(ddpg_params.policy_target, env, "policy_target.data")
            save_model(ddpg_params.value_estimator_target, env, "value_estimator_target.data")

            if scaler is not None:
                save_scaler(scaler, env, "scaler.data")

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


def ddpg_run(env, policy, scaler=None, render=True):
    episode_number = 0
    episode_rewards = []
    while True:
        state, done, episode_reward, episode_length = env.reset(), False, 0, 0
        while not done:
            if scaler:
                state = scale_state(scaler, state)

            action = policy.get_actions(torch.tensor(state).float())
            action = np.clip(action, env.action_space.low, env.action_space.high)

            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)

        print(f"Episode {episode_number}\t"
              f"Reward: {episode_reward:.3f}\t"
              f"Number of steps: {episode_length}\t"
              f"Avg reward: {np.mean(episode_rewards):.3f} +- {np.std(episode_rewards):.3f}")
        episode_number += 1
