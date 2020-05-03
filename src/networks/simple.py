from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, uniform_


class SimplePolicyDiscrete(nn.Module):
    """ Simple policy for discrete actions inspired by Torch's example policy for the Reinforce example code.
    See: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """

    def __init__(self, input_size: int, output_size: int):
        super(SimplePolicyDiscrete, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, output_size)

        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)  # Use softmax to return probabilities


class SimplePolicyContinuous(nn.Module):
    """ Simple policy for continuous actions, using a Gaussian normal as the distribution from which
    to sample the actions. The parameters (mean, std) of the distribution are computed from the state
    using a neural network. The architecture is taken from:
    https://github.com/lantunes/mountain-car-continuous/blob/master/rl/reinforce/agent.py
    """

    def __init__(self, input_size: int, output_size: int):
        super(SimplePolicyContinuous, self).__init__()
        self.output_size = output_size

        self.affine1Mu = nn.Linear(input_size, 128)
        self.affine1Mu.bias.data.fill_(0)
        self.affine2Mu = nn.Linear(128, 128)
        self.affine2Mu.bias.data.fill_(0)
        self.affine3Mu = nn.Linear(128, output_size, bias=False)
        self.affine3Mu.weight.data.fill_(0)

        # Important: It must be a Parameter and not a variable! If it's a variable
        # then it won't be part of the parameters given to the optimizer, meaning
        # it will never change. This will mean sigma will never improve and so
        # the results will stay very bad.
        self.hiddenSigma = Parameter(torch.zeros(32), requires_grad=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.affine1Mu(x)
        mu = torch.tanh(mu)
        mu = self.affine2Mu(mu)
        mu = torch.tanh(mu)
        mu = self.affine3Mu(mu)

        """
        This network has decided that the sigma will not depend on the state. 
        A possible question one might be asking is: why is there the exp() call below?
        I found the answer in the Spinning Up website by OpenAI:
        
            "Note that in both cases we output log standard deviations instead of standard deviations directly. 
            This is because log stds are free to take on any values in (-oo, oo), while stds must be nonnegative. 
            It’s easier to train parameters if you don’t have to enforce those kinds of constraints. The standard 
            deviations can be obtained immediately from the log standard deviations by exponentiating them, so 
            we do not lose anything  by representing them this way."
            
        Src: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
        """
        sigma = self.hiddenSigma.sum()
        sigma = torch.exp(sigma)

        return mu, sigma


class SimpleCritic(nn.Module):
    """ Same architecture as in the Pytorch example file for the Actor Critic method
    See: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
    """

    def __init__(self, input_size: int):
        super(SimpleCritic, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.affine1(x))
        state_value = self.value_head(x)
        return state_value


class SimplePolicyContinuous2(nn.Module):
    """
    Variant of SimplePolicyContinuous with a simpler architecture, taken from
    https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
    """

    def __init__(self, input_size: int, output_size: int):
        super(SimplePolicyContinuous2, self).__init__()

        self.affine1Mu = nn.Linear(input_size, 40)
        self.affine2Mu = nn.Linear(40, 40)
        self.affine3Mu = nn.Linear(40, output_size)
        self.affine1Sigma = nn.Linear(input_size, 1)

        xavier_uniform_(self.affine1Mu.weight)
        xavier_uniform_(self.affine2Mu.weight)
        xavier_uniform_(self.affine3Mu.weight)
        xavier_uniform_(self.affine1Sigma.weight)
        self.affine1Mu.bias.data.fill_(0)
        self.affine2Mu.bias.data.fill_(0)
        self.affine3Mu.bias.data.fill_(0)
        self.affine1Sigma.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.affine1Mu(x)
        mu = F.elu(mu)
        mu = self.affine2Mu(mu)
        mu = F.elu(mu)
        mu = self.affine3Mu(mu)

        sigma = self.affine1Sigma(x)
        sigma = F.softplus(sigma) + 1e-5

        return mu, sigma


class SimpleCritic2(nn.Module):
    """
    Variant of SimpleCriticContinuous with a simpler architecture, taken from
    https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
    """

    def __init__(self, input_size: int):
        super(SimpleCritic2, self).__init__()

        self.affine1 = nn.Linear(input_size, 400)
        self.affine2 = nn.Linear(400, 400)
        self.affine3 = nn.Linear(400, 1)

        xavier_uniform_(self.affine1.weight)
        xavier_uniform_(self.affine2.weight)
        xavier_uniform_(self.affine3.weight)
        self.affine1.bias.data.fill_(0)
        self.affine2.bias.data.fill_(0)
        self.affine3.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.affine1(x)
        value = F.elu(value)
        value = self.affine2(value)
        value = F.elu(value)
        value = self.affine3(value)

        return value


# The following code and architecture was heavily inspired from the Spinning Up repository by OpenAI
# Source: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/core.py


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


class DDPGPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_high: np.ndarray, action_low: np.ndarray):
        super().__init__()
        self.layers = make_network([state_dim, 256, 256, action_dim], nn.ReLU, nn.Tanh, initialize_last_linear=True)
        self.action_high = torch.tensor(action_high)
        self.action_low = torch.tensor(action_low)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """ Returns the actions as a torch Tensor (gradients can be computed)"""
        actions = self.layers.forward(states)
        actions = actions * (self.action_high - self.action_low) + (self.action_high + self.action_low)
        return 0.5 * actions

    def get_actions(self, state: torch.Tensor) -> np.ndarray:
        """ Returns the actions as a Numpy array (no gradients will be computed) """
        with torch.no_grad():
            return self.forward(state).numpy()


class DDPGValueEstimator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.layers = make_network([state_dim + action_dim, 256, 256, 1], nn.ReLU, initialize_last_linear=True)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ Returns the actions as a torch Tensor (gradients can be computed)"""
        # Tensor are concatenated over the last dimension (e.g. the values, not the batch rows)
        full_input = torch.cat([states, actions], dim=-1)
        return self.layers.forward(full_input).squeeze(-1)


# Sigma must be in (e^-20, e^2) = (2e-9, 7.38)
LOG_SIGMA_MIN = -20
LOG_SIGMA_MAX = 2


class SacPolicy(nn.Module):
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

    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.layers = make_network([state_dim, 256, 256], nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(256, action_dim)
        self.sigma_layer = nn.Linear(256, action_dim)

        self.action_low = torch.tensor(action_low)
        self.action_high = torch.tensor(action_high)

    def forward(self, states: torch.Tensor, deterministic=False, compute_log_prob=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns the actions and their log probs as a torch Tensors (gradients can be computed)"""
        hidden_state = self.layers.forward(states)
        mu = self.mu_layer(hidden_state)
        log_std = self.sigma_layer(hidden_state)
        log_std = LOG_SIGMA_MIN + 0.5 * (LOG_SIGMA_MAX - LOG_SIGMA_MIN) * (torch.tanh(log_std) + 1)
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
        actions = action_center + actions * action_range
        return actions, log_prob

    def get_actions(self, state: torch.Tensor, deterministic=False, compute_log_prob=False) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """ Returns the actions as a Numpy array (no gradients will be computed) and optionally its log probability """
        with torch.no_grad():
            actions, log_prob = self.forward(state, deterministic, compute_log_prob)
            if compute_log_prob:
                return actions.numpy(), log_prob.numpy()
            else:
                return actions.numpy()


class SacValueEstimator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.layers = make_network([state_dim + action_dim, 256, 256, 1], nn.ReLU)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ Returns the actions as a torch Tensor (gradients can be computed)"""
        # Tensor are concatenated over the last dimension (e.g. the values, not the batch rows)
        full_input = torch.cat([states, actions], dim=-1)
        return self.layers.forward(full_input).squeeze(-1)
