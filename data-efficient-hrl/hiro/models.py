from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def var(tensor: torch.Tensor) -> Variable:
    if torch.cuda.is_available():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x, g) -> torch.Tensor:
        x = F.relu(self.l1(torch.cat([x, g], 1)))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


#
# class Critic(nn.Module):
#     def __init__(self, state_dim, goal_dim, action_dim):
#         super(Critic, self).__init__()
#
#         self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, 1)
#
#
#     def forward(self, x, g, u):
#         x = F.relu(self.l1(torch.cat([x, g, u], 1)))
#         x = F.relu(self.l2(x))
#         x = self.l3(x)
#         return x
#

class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, g, u) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([x, g, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, g, u) -> torch.Tensor:
        xu = torch.cat([x, g, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class ControllerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=1):
        super(ControllerActor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        self.scale = nn.Parameter(torch.tensor(scale).float(), requires_grad=False)
        self.actor = Actor(state_dim, goal_dim, action_dim, 1)

    def forward(self, x, g) -> torch.Tensor:
        return self.scale * self.actor(x, g)


class ControllerCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(ControllerCritic, self).__init__()

        self.critic = Critic(state_dim, goal_dim, action_dim)

    def forward(self, x, sg, u) -> torch.Tensor:
        return self.critic(x, sg, u)

    def Q1(self, x, sg, u) -> torch.Tensor:
        return self.critic.Q1(x, sg, u)


class ManagerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=None):
        super(ManagerActor, self).__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        self.scale = nn.Parameter(torch.tensor(scale).float(), requires_grad=False)
        self.actor = Actor(state_dim, goal_dim, action_dim, 1)

    def forward(self, x, g) -> torch.Tensor:
        return self.scale * self.actor(x, g)


class ManagerCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(ManagerCritic, self).__init__()
        self.critic = Critic(state_dim, goal_dim, action_dim)

    def forward(self, x, g, u) -> torch.Tensor:
        return self.critic(x, g, u)

    def Q1(self, x, g, u) -> torch.Tensor:
        return self.critic.Q1(x, g, u)
