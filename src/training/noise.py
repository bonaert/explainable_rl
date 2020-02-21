import copy
import numpy as np


class NormalNoise:
    """ Generates noise from a standard Normal N(0, 1) distribution """

    def __init__(self, size, decay=1):
        """ Initialize parameters and noise process. The noise can be decayed over time """
        self.size = size
        self.decay = decay
        self.current_decay = 1

    def reset(self):
        """ Resets the environment. Has no effect for Normal / Gaussian noise, since there is no state. """
        pass

    def update_noise_coeff(self):
        """ Grows the decay by multiplying the current decay by the decay coefficient """
        self.current_decay *= self.decay

    def sample(self):
        """ Update internal state and return it as a noise sample """
        return self.current_decay * np.random.randn(self.size)


# Src: https://github.com/amuta/DDPG-MountainCarContinuous-v0/blob/master/OUNoise.py
class OUNoise:
    """ Generates noise from a Ornstein-Uhlenbeck noise process """

    def __init__(self, size=1, mu=0, theta=0.05, sigma=0.25, decay=1):
        """Initialize parameters and noise process. The noise can be decayed over time. """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.decay = decay
        self.current_decay = 1
        self.reset()

    def reset(self):
        """ Reset the internal state (= noise) to mean (mu) """
        self.state = copy.copy(self.mu)

    def update_noise_coeff(self):
        """ Grows the decay by multiplying the current decay by the decay coefficient """
        self.current_decay *= self.decay

    def sample(self):
        """ Update internal state and return it as a noise sample """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = self.current_decay * (x + dx)
        return self.state
