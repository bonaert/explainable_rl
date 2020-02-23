#####################
# Watershed problem #
#####################

# The original watershed problem is a continuous optimization problem.
# In this problem, we can control the flow of water and attempt to maximize
# a scoring function.

# Here's a visualisation of the environment:
#
#                                       x1
#                                  ------------> City
#                                 /                                           x5
#                                /                   x2                ----------------> Farms
#  Upper river   Q1 = 92 --------------> Dam ----------------         /
#                                         S                  |       /
#                                                            |-------------------------> Downstream
#                                                            |                x6
#  Lower river   Q2 = 39 ------------------------------------
#                                \                   x3
#                                 \
#                                  -------------> Farms
#                                       x4
#
# The dam can store water.

# The environment is defined by:
# Q1: the amount of water arriving at the upper river (L^3)
# Q2: the amount of water arriving at the lower river (L^3)
# S:  the storage capacity of the dam

# We are able to modify the value of x1, x2, x4 and x6
# The values of x3 and x5 are deduced from the values of Q and x1, x2, x4 and x6
# x3 = Q2 - x4
# x5 = x2 + x3 - x6


# Some modifications were needed to transform this continuous optimization
# problem into a reinforcement learning (RL) problem.
# Here is the final scheme that was adopted.


# State
# ------
# The state of the environment is defined by the values of x1, x2, x3, x4, x5 and x6
# The values of Q1, Q2 and S are not changed during the episodes
# They are only set when creating or resetting an environment


# Actions
# -------
# The action is a continuous action space of dimension 4.
# The values of x1, x2, x4 and x6 have lower and upper bounds given by the equations:
#
# Note: we use the mathematical notation below
#
#     alpha1  <=  x1  <= Q1 - alpha2
#     0       <=  x2  <= S + Q1 - alpha1
#     alpha3  <=  x4  <= Q2 - alpha4
#     alpha5  <=  x6  <= S + Q1 + Q2 - alpha1 - alpha3 - alpha6
#
# Thus, for each variable x, we may increase it at most by MAX - MIN or
# decrease it at most by MAX - MIN. Thus, the action range for each
# variable x is [-(MAX - MIN), (MAX - MIN)]
# When applying an action, we ensure that the resulting state is still within
# bounds, clipping it inside the bounds if necessary.


# Reward
# ---------
#
# At each step, the agent is able to modify the values of x1, x2, x4 and x6
# The reward is given by an fitness function:
#     f = objective - violationPenalty
#
# Objective
# ---------
# For each variable x1, x2, x3, x4, x5 and x6, we compute its contribution
# to the objective. The equation for variable x is of the form
#
#      V = ax^2 + bx + c
#
# The total objective value is the sum of the contributions of all variables.
#
# Violations
# ----------
# There are 9 constraints which might be violated.
# If all constraints are respected, then violationPenalty = 0
# Otherwise, if n constraints are violated, then violationPenalty = C * (n + 1)
# where C is a constant


# Stopping point
# --------------
# There isn't a clear stopping point where success has been achieved.
# Instead, we define the number of episodes during we can take actions.
# The current value, taken from the paper, is defined to be 1000.


import os
import random
from typing import Union, List

import gym
import numpy as np
import torch

from gym import error, spaces, utils
from gym.utils import seeding

# Reading the scenarios from the files
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


def read_flows_from_file(filename: str):
    """ Read values in the file specified by the path.
    There is one number per line. The files end with a new line"""
    with open(os.path.join(DATA_DIR, filename)) as f:
        return [int(x.strip()) for x in f.readlines() if x.strip()]


all_Q1 = read_flows_from_file("Q1_proportional.txt")
all_Q2 = read_flows_from_file("Q2_proportional.txt")
all_S = read_flows_from_file("S_proportional.txt")

limited_Q1 = [160, 115, 80]
limited_Q2 = [65, 50, 35]
limited_S = [15, 12, 10]


class WatershedEnv(gym.Env):
    """ Watershed environment that respects the OpenAI gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, limited_scenarios: bool = False, increment_actions: bool = False, bizarre_states: bool = False):
        """ Creates the Watershed environment
        :param limited_scenarios: if True, use only the 3 scenarios in the paper instead of the whole 150 scenarios
        :param increment_actions: if True, actions increment the state. If False, the state is overwritten by the action
        :param bizarre_states: if False, use the values of the direct variables as the state. If False, just use the
        scenario number as the state (constant during a whole episode).
        """
        super(WatershedEnv, self).__init__()

        self.relevant_q1 = limited_Q1 if limited_scenarios else all_Q1
        self.relevant_q2 = limited_Q2 if limited_scenarios else all_Q2
        self.relevant_s = limited_S if limited_scenarios else all_S
        self.number_of_scenarios = len(self.relevant_q1)

        self.bizarre_states = bizarre_states
        self.increment_actions = increment_actions

        # Internal details
        self.step_number = 0
        self.step_number_one_hot = None
        self.total_number_of_episodes = 1000 if increment_actions else 50

        # Fitness
        self.previous_fitness = None
        self.fitness = 0

        # Objective function
        self.objective_scores = [0.0] * 6
        self.total_objective_score = 0

        # Violations
        self.constraint_values = [0.0] * 9
        self.violations = 0
        self.violation_penalty = 0
        self.num_violations = 0
        self.violations_multiplier = 100

        self.total_violations_sum = 0

        # Objective function coefficients
        self.a = [-0.20, -0.06, -0.29, -0.13, -0.056, -0.15]
        self.b = [6.0, 2.5, 6.28, 6.0, 3.74, 7.6]
        self.c = [-5.0, 0.0, -3.0, -6.0, -23.0, -15.0]

        # Used in verifying boundaries and computing constraints
        self.alpha = [12.0, 10.0, 8.0, 6.0, 15.0, 10.0]

        self.setup_environment_parameters()

        # Variables to be optimized
        self.x = np.zeros(6)  # To make pylint happy and get better auto-completion
        self.reinitialise_state()

    def setup_environment_parameters(self):
        # "Q1 and Q2 represent the monthly inflows of water into respectively
        # the mainstream and tributary (in L^3)"
        self.scenario_number = random.randint(0, self.number_of_scenarios - 1)
        self.step_number_one_hot = np.zeros(self.number_of_scenarios)
        self.step_number_one_hot[self.scenario_number] = 1

        self.Q1 = self.relevant_q1[self.scenario_number]
        self.Q2 = self.relevant_q2[self.scenario_number]
        # "S is storage capacity of the dam (in L^3)"
        self.S = self.relevant_s[self.scenario_number]
        # Lower and upper bounds for each variable that is directly
        # controllable by the ser.
        self.flows_lower_bounds = np.array([
            self.alpha[0],  # x1
            0,  # x2
            self.alpha[2],  # x4
            self.alpha[4]  # x6
        ])
        self.flows_upper_bounds = np.array([
            self.Q1 - self.alpha[1],  # x1
            self.S + self.Q1 - self.alpha[0],  # x2
            self.Q2 - self.alpha[3],  # x4
            self.S + self.Q1 + self.Q2 - self.alpha[0] - self.alpha[2] - self.alpha[5]  # x6
        ])

        # Define action and observation space
        # Observation space = 6-dimensional real values (values for x1, x2, x3, x4, x5, x6)
        if not self.bizarre_states:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6 + self.number_of_scenarios, 1)
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.number_of_scenarios, 1)
            )

        # Action space = 4-dimensional real values (maximum allowed changes in x1, x2, x4, x6)
        if self.increment_actions:
            self.action_space = spaces.Box(
                low=-np.ones(4),
                high=np.ones(4)
            )
        else:
            self.action_space = spaces.Box(
                low=self.flows_lower_bounds,
                high=self.flows_upper_bounds
            )

    def reinitialise_state(self):
        initial_random_values = np.random.uniform(low=self.flows_lower_bounds, high=self.flows_upper_bounds)
        self.update_all_flows(initial_random_values)
        self.previous_fitness = None
        self.fitness = 0

    def update_violations(self):
        self.constraint_values[0] = self.alpha[0] - self.x[0]
        self.constraint_values[1] = self.alpha[1] - self.Q1 + self.x[0]
        self.constraint_values[2] = self.x[1] - self.S - self.Q1 + self.x[0]
        self.constraint_values[3] = self.alpha[3] - self.x[2]
        self.constraint_values[4] = self.alpha[2] - self.x[3]
        self.constraint_values[5] = self.alpha[3] - self.Q2 + self.x[3]
        self.constraint_values[6] = self.alpha[5] - self.x[4]
        self.constraint_values[7] = self.alpha[4] - self.x[5]
        self.constraint_values[8] = self.alpha[5] - self.x[1] - self.x[2] + self.x[5]

        violationsList = [v for v in self.constraint_values if v > 0]
        self.violations = sum(violationsList)
        self.num_violations = len(violationsList)
        if self.violations > 0:
            # The penalty function for a constrain violation is given by C * (V + 1)
            # Where C is the violation constant and V the violation amount
            self.violation_penalty = self.violations_multiplier * (self.violations + 1)
        else:
            self.violation_penalty = 0  # = violations

    def update_objective(self):
        for i in range(6):
            self.objective_scores[i] = self.a[i] * self.x[i] ** 2 + self.b[i] * self.x[i] + self.c[i]

        self.total_objective_score = sum(self.objective_scores)

    def update_fitness(self):
        self.previous_fitness = self.fitness
        self.update_violations()
        self.update_objective()
        self.fitness = self.total_objective_score - self.violation_penalty

    def update_all_flows(self, flowsControllableByAgent: Union[List[float], np.ndarray]):
        # flowsControllableByAgent =
        #    Programming notation: [x[0], x[1], x[3], x[5]]
        #    Math notation:        [x1,   x2,   x4,   x6  ]

        # First update the direct variables
        self.x[0] = flowsControllableByAgent[0]
        self.x[1] = flowsControllableByAgent[1]
        self.x[3] = flowsControllableByAgent[2]
        self.x[5] = flowsControllableByAgent[3]

        # Then deduce the indirect variables
        self.x[2] = self.Q2 - self.x[3]  # x3 = Q2 - x4
        self.x[4] = self.x[1] + self.x[2] - self.x[5]  # x5 = x2 + x3 - x6

    def step(self, action: Union[np.ndarray, torch.Tensor]):
        """
        Returns observation, reward, done, info = env.step(action)
            - observation (object)
            - reward (float)
            - done (boolean)
            - info (dict): diagnostic information useful for debugging
        """
        if type(action) == torch.Tensor:
            action = action.squeeze().numpy()

        if not type(action) is np.ndarray:
            raise Exception("The action must be a Numpy array but is of type %s (value = %s)" % (type(action), action))

        if self.increment_actions and not self.action_space.contains(action):
            action = action.clip(self.action_space.low, self.action_space.high)

        # Additionally, we must make sure the value will stay in the range
        # min <= x + action <= max
        if self.increment_actions:
            current_values = self.x[np.array([0, 1, 3, 5])]
            new_flow_values = current_values + action
        else:
            new_flow_values = action

        new_flow_values = np.clip(new_flow_values, self.flows_lower_bounds, self.flows_upper_bounds)
        self.update_all_flows(new_flow_values)

        if any([x < 0 for x in self.x]):
            pass
            # TODO: should I clip the actions to ensure the flows are always positive?
            # raise Exception(f"Negative flows! x = {[round(x, 4) for x in self.x]}")

        self.update_fitness()

        self.step_number += 1

        # reward = self.fitness - self.previous_fitness
        reward = self.fitness
        if self.bizarre_states:
            observation = self.step_number_one_hot
        else:
            observation = np.hstack([self.step_number_one_hot, self.x])

        done = (self.step_number == self.total_number_of_episodes)
        info = {}
        return observation, reward, done, info

    def reset(self):
        """ Reset the environment to its initial values """
        self.step_number = 0

        self.setup_environment_parameters()

        # Reset state to random values
        self.reinitialise_state()

        # Fitness
        self.fitness = 0

        # Objective function
        self.objective_scores = [0.0] * 6
        self.total_objective_score = 0

        # Violations
        self.constraint_values = [0.0] * 9
        self.violations = 0
        self.violation_penalty = 0
        self.num_violations = 0
        self.violations_multiplier = 100

        self.total_violations_sum = 0

        if self.bizarre_states:
            return self.step_number_one_hot
        else:
            return np.hstack([self.step_number_one_hot, self.x])

    def render(self, mode: str = 'human'):
        """ Renders the environment and the state of the flows """
        # Note, this may seem badly aligned but it is correct!
        # It becomes aligned after substituting the variables by their real value
        content = """
                          x1 = {x1:.2f}
                        ------------> City
                       /                                            x5 = {x5:.2f}
                      /                 x2 = {x2:.2f}              ----------------> Farms
        Q1 = {Q1} --------------> Dam ----------------         /
                              S = {S}                |       /
                                                    |-------------------------> Lower triburary
                                                    |             x6 = {x6:.2f}
        Q2 = {Q2} ------------------------------------
                      \\                 x3 = {x3:.2f}
                       \\
                        -------------> Farms
                         x4 = {x4:.2f}
        """
        print(
            content.format(
                Q1=self.Q1,
                Q2=self.Q2,
                S=self.S,
                x1=self.x[0],
                x2=self.x[1],
                x3=self.x[2],
                x4=self.x[3],
                x5=self.x[4],
                x6=self.x[5]
            )
        )

    def close(self):
        # Nothing to do in this environment
        pass
