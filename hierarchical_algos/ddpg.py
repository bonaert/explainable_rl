from typing import List

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam

from common import get_tensor, ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_size: int,
                 action_range: np.ndarray, action_center: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(state_size + goal_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.action_range = torch.from_numpy(action_range)
        self.action_center = torch.from_numpy(action_center)
        self.has_goal = goal_size > 0

    def forward(self, state: np.ndarray, goal: np.ndarray) -> torch.Tensor:
        if self.has_goal:
            state, goal = get_tensor(state), get_tensor(goal)
            total_input = torch.cat([state, goal], dim=-1)  # Concatenate to format [states | goals]
        else:
            total_input = get_tensor(state)

        x = self.fc1(total_input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return self.action_center + self.action_range * torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_size: int, q_bound: float):
        super().__init__()
        self.fc1 = nn.Linear(state_size + goal_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.has_goal = goal_size > 0
        self.q_bound = q_bound
        self.has_q_bound = (q_bound is not None)

    def forward(self, state: np.ndarray, goal: np.ndarray, action: np.ndarray) -> torch.Tensor:
        if self.has_goal:
            state, goal, action = get_tensor(state), get_tensor(goal), get_tensor(action)
            total_input = torch.cat([state, goal, action], dim=-1)  # Concatenate to format [states | goals | actions]
        else:
            state, action = get_tensor(state), get_tensor(action)
            total_input = torch.cat([state, action], dim=-1)  # Concatenate to format [states | actions]

        x = self.fc1(total_input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        if self.has_q_bound:
            return self.q_bound * torch.sigmoid(x)
        else:
            return x


class DDPG(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_range: np.ndarray, action_center: np.ndarray, q_bound: float,
                 buffer_size: int, batch_size: int):
        super().__init__()
        self.action_size = len(action_range)

        # Important: there are no target networks on purpose, because the HAC paper
        # found they were not very useful
        self.critic = Critic(state_size, goal_size, self.action_size, q_bound)
        self.actor = Actor(state_size, goal_size, self.action_size, action_range, action_center)

        # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/master/critic.py#L8
        self.critic_optimizer = Adam(self.critic.parameters(), lr=0.001)
        # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/master/actor.py#L15
        self.actor_optimizer = Adam(self.actor.parameters(), lr=0.001)

        # 8 transitions dims: (current_state, action, env_reward, total_reward, next_state, transition_reward, current_goal, discount)
        # NOTE: they use some more complicated logic (which depends on the level) to determinate the size of the buffer
        # TODO: this is a simplfication. See if it works anyway.
        self.buffer = ReplayBuffer(buffer_size, num_transition_dims=8)
        self.batch_size = batch_size
        self.q_bound = q_bound

    def add_to_buffer(self, transition: tuple):
        assert len(transition[1]) == self.action_size
        self.buffer.add(transition)

    def add_many_to_buffer(self, transitions: List[tuple]):
        self.buffer.add_many(transitions)

    def sample_action(self, state: np.ndarray, goal: np.ndarray, **kwargs) -> np.ndarray:
        with torch.no_grad():
            return self.actor(state, goal).numpy()

    def learn(self, num_updates: int):
        # If there's not enough transitions to fill a batch, we don't do anything
        if self.buffer.size() < self.batch_size:
            return

        for i in range(num_updates):
            # Step 1: get the transitions and the next actions for the next state
            states, actions, env_rewards, total_env_rewards, next_states, rewards, goals, discounts = self.buffer.get_batch(self.batch_size)
            next_actions = self.actor(next_states, goals)

            # Step 2: improve the critic
            with torch.no_grad():  # No need to compute gradients for this
                target_q_values = rewards + discounts * self.critic(next_states, goals, next_actions).squeeze()
                target_q_values = target_q_values.unsqueeze(1)
                # We clamp the Q-values to be in [-H, 0] if we're not at the top level. Why would this be needed given that the critic already
                # outputs values in this range? Well, it's true, the critic does do that, but the target
                # expression is r + alpha * Q(s', a') and that might go outside of [-H, 0]. We don't want
                # that to happen, so we clamp it to the range. This will thus incentivize the critic to predict
                # values in [-H, 0], but since the critic can already only output values in that range, it's perfect.
                # Of course, this is not a coincidence but done by design.
                if self.q_bound is not None:  # It's None for the top-level, since we don't know in advance the total reward range
                    target_q_values = torch.clamp(target_q_values, min=self.q_bound, max=0.0)

            self.critic_optimizer.zero_grad()
            predicted_q_values = self.critic(states, goals, actions)
            critic_loss = F.mse_loss(predicted_q_values, target_q_values)
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
            self.actor_optimizer.zero_grad()
            new_actions = self.actor(states, goals)
            actor_loss = -self.critic(states, goals, new_actions).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze Q-network so you can optimize it at next DDPG step.
            # for p in self.critic.parameters():
            #     p.requires_grad = True