import itertools
from collections import deque
from dataclasses import dataclass

import gym
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from networks.simple import DDPGPolicy, DDPGValueEstimator
from training.common import RunParams, TrainingInfo, setup_scaler, scale_state, log_on_console, log_on_tensorboard, \
    close_tensorboard


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

    start_steps: int  # During the first start_steps steps, we pick random actions


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

    return ((values - values_expected) ** 2).mean()


def update_models(batch_transitions, ddpg_params: DDPGParams, run_params: RunParams, writer: SummaryWriter, step_number):
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
                param_target.data.add_((1 - ddpg_params.polyak) * param)


def select_action_ddpg(state, ddpg_params: DDPGParams, env: gym.Env) -> np.ndarray:
    action = ddpg_params.policy.get_actions(torch.tensor(state))
    action += ddpg_params.noise_coeff * np.random.randn(env.action_space.shape[0])  # Gaussian noise
    action = np.clip(action, env.action_space.low, env.action_space.high)  # Clamp the action inside the action space
    return action


def ddpg_train(
        env: gym.Env,
        run_params: RunParams,
        ddpg_params: DDPGParams):
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
    if run_params.should_scale_states:
        scaler = setup_scaler(env)
    training_info = TrainingInfo(GAMMA=run_params.gamma)
    replay_buffer = ReplayBuffer(ddpg_params.replay_buffer_size)

    step_number = 0
    max_episode_steps = env.spec.max_episode_steps

    for episode_number in itertools.count():  # itertools.count() is basically range(+infinity)
        state = env.reset()

        # Do a whole episode (upto 10000 steps, don't want infinite steps)
        for t in range(max_episode_steps):
            if run_params.should_scale_states:
                state = scale_state(scaler, state)

            # Pick an action, execute and observe the results
            # Note: in the first start_steps steps, we randomly pick actions from
            # the action space (uniformly) to have better exploration.
            if t > ddpg_params.start_steps:
                action = select_action_ddpg(state, ddpg_params, env)
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)

            # Render the environment if wanted
            if run_params.should_render(episode_number):
                env.render()

            # Store reward and updates the running reward
            training_info.record_step(state, action, reward)

            # Add the transition to the replay buffer
            new_state_scaled = scale_state(scaler, state) if run_params.should_scale_states else new_state
            replay_buffer.store(state, action, reward, new_state_scaled, done and t < max_episode_steps - 1)

            state = new_state
            if done:
                break

            if step_number >= ddpg_params.update_start and step_number % ddpg_params.update_frequency == 0:
                for update_step in range(ddpg_params.update_frequency):
                    batch_transitions = replay_buffer.sample_batch(ddpg_params.batch_size)
                    update_models(batch_transitions, ddpg_params, run_params, writer, step_number)

            step_number += 1

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

    close_tensorboard(run_params, writer)
