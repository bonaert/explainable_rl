from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SimplePolicyDiscrete(nn.Module):
    """ Simple policy for discrete actions inspired by Torch's example policy for the Reinforce example code.
    See: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """

    def __init__(self, input_size: int, output_size: int):
        super(SimplePolicyDiscrete, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, output_size)

    def forward(self, x):
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
        self.hiddenSigma = Parameter(torch.zeros(32))

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.affine1Mu(x)
        mu = torch.tanh(mu)
        mu = self.affine2Mu(mu)
        mu = torch.tanh(mu)
        mu = self.affine3Mu(mu)  # No bias, raw matrix multiplication

        sigma = self.hiddenSigma.sum()
        sigma = torch.exp(sigma)

        return mu, sigma
