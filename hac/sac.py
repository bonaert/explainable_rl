import copy
import random
from typing import List, Union, Tuple, Optional

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn.init import uniform_
from torch.optim import Adam
from common import get_tensor, ReplayBuffer, polyak_average, permissive_get_tensor
from priorited_buffer import PrioritizedReplayBuffer


def make_network(layer_sizes: List[int], activation: torch.nn.Module, output_activation: torch.nn.Module = nn.Identity,
                 initialize_last_linear: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(layer_sizes) - 1):
        size_before, size_after = layer_sizes[i], layer_sizes[i + 1]
        linear_layer = nn.Linear(size_before, size_after)
        layers.append(linear_layer)

        if i == len(layer_sizes) - 2 and initialize_last_linear:
            uniform_(linear_layer.weight.data, -0.003, 0.003)
            uniform_(linear_layer.bias.data, -0.003, 0.003)

        layers.append(activation() if i < len(layer_sizes) - 2 else output_activation())
    return nn.Sequential(*layers)


# Sigma must be in (e^-20, e^2) = (2e-9, 7.38)
# Sigma must be in (e^-20, e^2) = (0.01, 7.38)
LOG_SIGMA_MIN = -10
LOG_SIGMA_MAX = 2


class SacActor(nn.Module):
    """
    Design decisions:
    1) Sample actions from a Normal distribution. This has 2 advantages:
        - We can compute the entropy of our policy, which is required in the SAC algorithm
        - We get automatic noise (because we sample from a distribution). If needed, we can however still
          get deterministic actions by just returning mu. This is useful at test time (pure exploitation)
    2) The mean and std of the standard distribution are obtained through neural networks.
       Both networks take the state as input. According to Spinning up, if sigma didn't depend on the
       state, SAC wouldn't work. They challenge us to figure out why and to test this claim empirically.
    """

    def __init__(self, state_size: int, goal_size: int, action_size: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.layers = make_network([state_size + goal_size, 256, 256], activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(256, action_size)
        self.sigma_layer = nn.Linear(256, action_size)

        self.action_low = torch.from_numpy(action_low)
        self.action_high = torch.from_numpy(action_high)

        self.has_goal = (goal_size > 0)

    def forward(self, state: np.ndarray, goal: np.ndarray, deterministic=False, compute_log_prob=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns the actions and their log probs as a torch Tensors (gradients can be computed)"""
        if self.has_goal:
            state, goal = get_tensor(state), get_tensor(goal)
            total_input = torch.cat([state, goal], dim=-1)  # Concatenate to format [states | goals]
        else:
            total_input = get_tensor(state)

        hidden_state = self.layers.forward(total_input)
        mu = self.mu_layer(hidden_state)
        log_std = self.sigma_layer(hidden_state)
        log_std = LOG_SIGMA_MIN + (LOG_SIGMA_MAX - LOG_SIGMA_MIN) * (torch.tanh(log_std) + 1) / 2.0
        # log_std = torch.clamp(log_std, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        std = torch.exp(log_std)

        policy_distribution = Normal(mu, std)
        actions = mu if deterministic else policy_distribution.rsample()

        if compute_log_prob:
            # Exact source: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L54
            # "Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)"
            log_prob = policy_distribution.log_prob(actions).sum(axis=-1)
            try:
                log_prob -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
            except IndexError:
                log_prob -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum()
        else:
            log_prob = None

        actions = torch.tanh(actions)  # The log_prob above takes into account this "tanh squashing"
        action_center = (self.action_high + self.action_low) / 2
        action_range = (self.action_high - self.action_low) / 2
        actions_in_range = action_center + actions * action_range

        # print(f"Mu {mu}\t sigma {std}\tactions {actions}\taction_in_range {actions_in_range}")
        return actions_in_range, log_prob

    def sample_actions(self, state: np.ndarray, goal: np.ndarray, deterministic=False, compute_log_prob=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ Returns the actions as a Numpy array (no gradients will be computed) and optionally its log probability """
        with torch.no_grad():
            actions, log_prob = self.forward(state, goal, deterministic, compute_log_prob)
            if compute_log_prob:
                return actions.numpy(), log_prob.numpy()
            else:
                return actions.numpy()


class SacCritic(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_size: int, q_bound: Optional[float]):
        super().__init__()
        self.layers = make_network([state_size + goal_size + action_size, 256, 256, 1], activation=nn.ReLU)
        self.q_bound = q_bound
        self.has_goal = (goal_size > 0)
        self.has_q_bound = (q_bound is not None)

    def forward(self, state: np.ndarray, goal: np.ndarray, action: np.ndarray) -> torch.Tensor:
        """ Returns the actions as a torch Tensor (gradients can be computed)"""
        if self.has_goal:
            state, goal, action = get_tensor(state), get_tensor(goal), get_tensor(action)
            total_input = torch.cat([state, goal, action], dim=-1)  # Concatenate to format [states | goals | actions]
        else:
            state, action = get_tensor(state), get_tensor(action)
            total_input = torch.cat([state, action], dim=-1)  # Concatenate to format [states | actions]

        # Tensor are concatenated over the last dimension (e.g. the values, not the batch rows)
        x = self.layers.forward(total_input)
        if self.has_q_bound:
            return self.q_bound * torch.sigmoid(x)
        else:
            return x


class Sac(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_low: np.ndarray, action_high: np.ndarray, q_bound: float,
                 buffer_size: int, batch_size: int, writer, sac_id: Optional[str],
                 use_priority_replay: bool, learning_rate: float):
        super().__init__()
        self.action_size = len(action_low)
        self.use_priority_replay = use_priority_replay

        self.critic1 = SacCritic(state_size, goal_size, self.action_size, q_bound)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2 = SacCritic(state_size, goal_size, self.action_size, q_bound)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor = SacActor(state_size, goal_size, self.action_size, action_low=action_low, action_high=action_high)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic_optimizer = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=learning_rate)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)

        # Optimization for speed: don't compute gradients for the target networks, since we will never use them
        for network in [self.actor_target, self.critic1_target, self.critic2_target]:
            for parameter in network.parameters():
                parameter.requires_grad = False

        self.polyak = 0.995  # TODO: add this to params
        self.alpha = 0.01  # TODO: add this to params

        # 8 transitions dims: (current_state, action, env_reward, total_reward, next_state, transition_reward, current_goal, discount)
        # NOTE: they use some more complicated logic (which depends on the level) to determinate the size of the buffer
        # TODO: this is a simplfication. See if it works anyway.
        # self.buffer = PrioritizedReplayBuffer(buffer_size, num_transition_dims=8)

        if use_priority_replay:
            self.buffer = PrioritizedReplayBuffer(buffer_size, num_transition_dims=8)
        else:
            self.buffer = ReplayBuffer(buffer_size, num_transition_dims=8)

        self.batch_size = batch_size
        self.q_bound = q_bound

        self.step_number = 0
        self.use_tensorboard = (writer is not None)
        self.writer = writer
        self.sac_id = sac_id

    def get_error(self, transition: tuple) -> float:
        state, action, _, _, next_state, reward, goal, discount = [permissive_get_tensor(x) for x in transition]
        target_q_values, values1, values2 = self.get_target_q_values(reward, discount, next_state, goal)
        predicted_q_values1 = self.critic1.forward(state, goal, action)
        predicted_q_values2 = self.critic2.forward(state, goal, action)

        return self.get_td_error(predicted_q_values1, predicted_q_values2, target_q_values).item()

    def get_td_error(self, predicted_q_values1: torch.Tensor, predicted_q_values2: torch.Tensor, target_q_values: torch.Tensor) -> torch.Tensor:
        return (target_q_values - predicted_q_values1).abs() + (target_q_values - predicted_q_values2).abs()

    def add_to_buffer(self, transition: tuple):
        assert len(transition[1]) == self.action_size
        if self.use_priority_replay:
            # noinspection PyArgumentList
            self.buffer.add(error=self.get_error(transition), transition=transition)
        else:
            self.buffer.add(transition)

    def add_many_to_buffer(self, transitions: List[tuple]):
        for transition in transitions:
            self.add_to_buffer(transition)

    def sample_action(self, state: np.ndarray, goal: np.ndarray, deterministic=False) -> np.ndarray:
        with torch.no_grad():
            return self.actor.sample_actions(state, goal, deterministic, compute_log_prob=False)

    def learn(self, num_updates: int):
        # If there's not enough transitions to fill a batch, we don't do anything
        if self.buffer.size() < self.batch_size:
            return

        for i in range(num_updates):
            # Step 1: get the transitions and the next actions for the next state
            states, actions, env_rewards, total_env_rewards, next_states, rewards, goals, discounts = self.buffer.get_batch(self.batch_size)

            # Step 2: improve the critic
            target_q_values, values1, values2 = self.get_target_q_values(rewards, discounts, next_states, goals)
            predicted_q_values1 = self.critic1(states, goals, actions)
            predicted_q_values2 = self.critic2(states, goals, actions)

            # Update priority in Priority Replay Buffer if needed
            if self.use_priority_replay:
                errors = self.get_td_error(predicted_q_values1, predicted_q_values2, target_q_values)
                for j in range(self.batch_size):
                    index = self.buffer.last_indices[j]
                    self.buffer.update(index, errors[j].item())

            critic_loss = F.smooth_l1_loss(predicted_q_values1, target_q_values) + \
                          F.smooth_l1_loss(predicted_q_values2, target_q_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 3: improve the actor
            # Freeze Q-network so you don't waste computational effort
            # computing gradients for it during the policy learning step.
            # TODO: for some reason, if I do this, then I get this error when I do actor_loss.backward()
            # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
            # This does not happen in my other DDPG code and I don't know why.
            # TODO: figure it out
            # for p in self.critic.parameters():
            #     p.requires_grad = False

            # We want to maximize the q-values of the actions (and therefore minimize -Q_values)
            new_actions, log_new_actions = self.actor(states, goals)
            values1 = self.critic1(states, goals, new_actions)
            values2 = self.critic2(states, goals, new_actions)
            actor_loss = (self.alpha * log_new_actions - torch.min(values1, values2)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Log things on tensorboard and console if needed
            if self.use_tensorboard and i == 0:
                self.writer.add_scalar(f"Loss/({self.sac_id}) Policy", actor_loss.item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Value", critic_loss.item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Log Prob", log_new_actions[0].item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Target", target_q_values[0].item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Predicted 1", predicted_q_values1[0].item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Values 1", values2[0].item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Predicted 2", predicted_q_values2[0].item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Values 2", values1[0].item(), self.step_number)
                self.writer.add_scalar(f"Loss/({self.sac_id}) Reward", rewards[0].item(), self.step_number)

            # Unfreeze Q-network so you can optimize it at next DDPG step.
            # for p in self.critic.parameters():
            #     p.requires_grad = True

            polyak_average(self.actor, self.actor_target, self.polyak)
            polyak_average(self.critic1, self.critic1_target, self.polyak)
            polyak_average(self.critic2, self.critic2_target, self.polyak)

            self.step_number += 1

    def get_target_q_values(self, rewards: torch.Tensor, discounts: torch.Tensor, next_states: torch.Tensor, goals: torch.Tensor):
        with torch.no_grad():  # No need to compute gradients for this
            # The actions for the next state come from **current** policy (not from the target policy)
            next_actions, log_next_actions = self.actor(next_states, goals)

            values1 = self.critic1_target(next_states, goals, next_actions)
            values2 = self.critic2_target(next_states, goals, next_actions)
            target_q_values = rewards + discounts * (torch.min(values1, values2).squeeze() - self.alpha * log_next_actions)
            if target_q_values.ndim != 0:
                target_q_values = target_q_values.unsqueeze(1)
            # We clamp the Q-values to be in [-H, 0] if we're not at the top level. Why would this be needed given that the critic already
            # outputs values in this range? Well, it's true, the critic does do that, but the target
            # expression is r + alpha * Q(s', a') and that might go outside of [-H, 0]. We don't want
            # that to happen, so we clamp it to the range. This will thus incentivize the critic to predict
            # values in [-H, 0], but since the critic can already only output values in that range, it's perfect.
            # Of course, this is not a coincidence but done by design.
            if self.q_bound is not None:  # It's None for the top-level, since we don't know in advance the total reward range
                target_q_values = torch.clamp(target_q_values, min=self.q_bound, max=0.0)

            return target_q_values, values1, values2
