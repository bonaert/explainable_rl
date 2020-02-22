import itertools
from typing import List

import gym
import torch
from torch.distributions import Normal, Categorical
from torch.optim.optimizer import Optimizer

from src.networks.simple import SimplePolicyDiscrete, SimplePolicyContinuous
from src.training.common import select_action_continuous, select_action_discrete, TrainingInfo, setup_observation_scaler, \
    scale_state, RunParams, log_on_console, log_on_tensorboard, close_tensorboard, save_model, save_scaler


def train_policy(optimizer: Optimizer, training_info: TrainingInfo, run_params: RunParams):
    """ Trains the policy using the policy gradient method, given the discounted rewards of the latest episode
    Entropy is also taken into account. Each new episode diminishes its importance by run_params.entropy_decay,
    such that the agent will explore at the beginning and tend to explore less and less over time. The agent is
    trained once on all the transitions of the episode (instead of training many times over mini-batches).
    """
    training_info.compute_discounted_rewards()

    # Compute the loss of the policy at each time step
    policy_losses = []
    for log_prob, discounted_reward, entropy in zip(training_info.log_probs, training_info.discounted_rewards, training_info.entropies):
        entropy_coeff = run_params.entropy_coeff * run_params.entropy_decay ** training_info.episode_number
        policy_losses.append(-(log_prob + entropy_coeff * entropy) * discounted_reward)

    # Optimize the policy
    optimizer.zero_grad()
    total_policy_loss = torch.cat(policy_losses).sum()
    total_policy_loss.backward()
    optimizer.step()

    # Reset the state of the episode
    training_info.reset()


def train_policy_batches(policy: SimplePolicyContinuous, optimizer: Optimizer,
                         training_info: TrainingInfo, run_params: RunParams):
    """ Trains the policy using the policy gradient method, given the discounted rewards of the latest episode
    Entropy is also taken into account. Each new episode diminishes its importance by run_params.entropy_decay,
    such that the agent will explore at the beginning and tend to explore less and less over time. The agent is
    trained many times over mini-batches of transitions of the episode (instead of being trained once on all
    transitions)"""
    training_info.compute_discounted_rewards()

    for (states, actions, discounted_rewards) in training_info.get_batches(run_params.batch_size):
        train_batch(policy, states, actions, discounted_rewards, optimizer, training_info.episode_number, run_params)

    training_info.reset()


def train_batch(policy: SimplePolicyContinuous, states: List[torch.Tensor], actions: List[torch.Tensor],
                discounted_rewards: List[torch.Tensor], optimizer: Optimizer, episode_number: int, run_params: RunParams):
    """ Trains the policy using the policy gradient method using a single mini-batch of transitions.
    Entropy is also taken into account. Each new episode diminishes its importance by run_params.entropy_decay,
    such that the agent will explore at the beginning and tend to explore less and less over time"""
    optimizer.zero_grad()

    policy_losses = []
    for (state, action, discounted_reward) in zip(states, actions, discounted_rewards):
        state = state.float().unsqueeze(0)

        if run_params.continuous_actions:
            mu, sigma = policy.forward(state)
            n = Normal(mu, sigma)
        else:
            probs = policy.forward(state)
            n = Categorical(probs)
        policy_losses.append(-(n.log_prob(action) + 0.99 ** episode_number * n.entropy()) * discounted_reward)

    total_policy_loss = torch.cat(policy_losses).sum()
    total_policy_loss.backward()
    optimizer.step()


def reinforceTraining(
        policy: SimplePolicyDiscrete,
        env: gym.Env,
        optimizer: Optimizer,
        run_params: RunParams):
    """ Trains the policy using the REINFORCE algorithm. Training is done at the end of each episode, and can
    be done either using all transitions at once, or over many training iterations over mini-batches of transitions.
    Both discrete and continuous actions spaces are supported. Several features can be optionally enabled:
    1) Scaling / normalizing the states / observations
    2) Logging training statistics on Tensorboard
    3) Render the environment periodically (pick render_frequency in the RunParams)
    4) Save the policy (and optionally the observation / state scaler) periodically (see RunParams.save_model_frequency)
    """
    training_info = TrainingInfo(GAMMA=run_params.gamma)
    print(f"The goal is a running reward of at least {env.spec.reward_threshold}.")

    scaler = setup_observation_scaler(env) if run_params.should_scale_states else None

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

            new_state, reward, done, _ = env.step(action)

            if run_params.should_render(episode_number):
                env.render()

            training_info.record_step(state, action, reward)  # Store reward and updates the running reward
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

        if run_params.train_with_batches:
            train_policy_batches(policy, optimizer, training_info, run_params)
        else:
            train_policy(optimizer, training_info, run_params)

        if run_params.should_save_model(episode_number):
            save_model(policy, env, "policy.data")

            if scaler is not None:
                save_scaler(scaler, env, "scaler.data")

    close_tensorboard(run_params, writer)


