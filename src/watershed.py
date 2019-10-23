#import gym


# from gym import error, spaces, utils
# from gym.utils import seeding

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
#                                                            |       /
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
# TODO!


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
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def readFlowsFromFile(filename):
    """ Read values in the file specified by the path.
    There is one number per line. The files end with a new line"""
    with open(os.path.join(DATA_DIR, filename)) as f:
        return [int(x.strip()) for x in f.readlines() if x.strip()]


allQ1 = readFlowsFromFile("Q1_proportional.txt")
allQ2 = readFlowsFromFile("Q2_proportional.txt")
allS = readFlowsFromFile("S_proportional.txt")

class WatershedEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, scenarioNum=0):
        # Internal details
        self.stepNum = 0
        self.TOTAL_NUMBER_OF_EPISODES = 1000

        # Variables to be optimized
        self.x = [0.0] * 6

        # Fitness
        self.fitness = 0

        # Objective function
        self.objectiveScores = [0.0] * 6
        self.totalObjectiveScore = 0

        # Violations
        self.constraintValues = [0.0] * 9
        self.violations = 0
        self.violationPenalty = 0
        self.numViolations = 0
        self.violationsMultiplier = 100

        self.totalViolationsSum = 0

        # Objective function coefficients
        self.a = [-0.20, -0.06, -0.29, -0.13, -0.056, -0.15]
        self.b = [6.0, 2.5, 6.28, 6.0, 3.74, 7.6]
        self.c = [-5.0, 0.0, -3.0, -6.0, -23.0, -15.0]

        # Used in verifying boundaries and computing constraints
        self.alpha = [12.0, 10.0, 8.0, 6.0, 15.0, 10.0]

        # TODO: implement reading these values from the files
        # "The Watershed problem evaluated in this paper will consist of a total of 150 flow
        # states for the river. This is divided up into training data (100 states) and
        # testing data (50 states)."

        # "Q1 and Q2 represent the monthly inflows of water into respectively
        # the mainstream and tributary (in L^3)"
        self.Q1 = allQ1[scenarioNum]
        self.Q2 = allQ2[scenarioNum]

        # "S is storage capacity of the dam (in L^3)"
        self.S = allS[scenarioNum]

        # Upper and lower bounds (the first 2 lines are there to remove warnings)
        self.lowerBounds = [0.0] * 6
        self.upperBounds = [0.0] * 6
        self.updateBounds()





    def updateBounds(self):
        # TODO: in his code, he creates boundaries arrays too, but he never seems to use
        # them except to do fractions <-> liters conversions. This seems strange to me,
        # since there's no guarantee that the values of x will be in the range. This may
        # be a bug in his code, or maybe these constraints are always true for a reason
        # I don't understand yet.

        # We don't have explicit constrains for x[2] and x[4] so we set NaN
        nan = float('NaN')
        self.lowerBounds = [self.alpha[0], 0, nan, self.alpha[2], nan, self.alpha[4]]
        self.upperBounds = [
            self.Q1 - self.alpha[1],
            self.S + self.Q1 - self.alpha[0],
            nan,
            self.Q2 - self.alpha[3],
            nan,
            self.S + self.Q1 + self.Q2 - self.alpha[0] - self.alpha[2] - self.alpha[5]
        ]

    def updateViolations(self):
        self.constraintValues[0] = self.alpha[0] - self.x[0]
        self.constraintValues[1] = self.alpha[1] - self.Q1 + self.x[0]
        self.constraintValues[2] = self.x[1] - self.S - self.Q1 + self.x[0]
        self.constraintValues[3] = self.alpha[3] - self.x[2]
        self.constraintValues[4] = self.alpha[2] - self.x[3]
        self.constraintValues[5] = self.alpha[3] - self.Q2 + self.x[3]
        self.constraintValues[6] = self.alpha[5] - self.x[4]
        self.constraintValues[7] = self.alpha[4] - self.x[5]
        self.constraintValues[8] = self.alpha[5] - self.x[1] - self.x[2] + self.x[5]

        violationsList = [v for v in self.constraintValues if v > 0]
        self.violations = sum(violationsList)
        self.numViolations = len(violationsList)
        if self.violations > 0:
            # The penalty function for a constrain violation is given by C * (V + 1)
            # Where C is the violation constant and V the violation amount
            self.violationPenalty = self.violationsMultiplier * (self.violations + 1)
        else:
            self.violationPenalty = 0  # = violations

    def updateObjective(self):
        for i in range(6):
            self.objectiveScores[i] = self.a[i] * self.x[i] ** 2 + self.b[i] * self.x[i] + self.c[i]

        self.totalObjectiveScore = sum(self.objectiveScores)

    def updateFitness(self):
        self.fitness = self.totalObjectiveScore - self.violationPenalty

    def updateAllFlows(self, flowsControllableByAgent):
        # flowsControllableByAgent =
        #    Programming notation: [x[0], x[1], x[3], x[5]]
        #    Math notation:        [x1,   x2,   x4,   x6  ]

        # First update the direct variables
        self.x[0] = flowsControllableByAgent[0]
        self.x[1] = flowsControllableByAgent[1]
        self.x[3] = flowsControllableByAgent[2]
        self.x[5] = flowsControllableByAgent[3]

        # Then deduce the indirect variables
        self.x[2] = self.Q2 - self.x[3]                # x3 = Q2 - x4
        self.x[4] = self.x[1] + self.x[2] - self.x[5]  # x5 = x2 + x3 - x6

    def step(self, action):
        """
        Returns observation, reward, done, info = env.step(action)
            - observation (object)
            - reward (float)
            - done (boolean)
            - info (dict): diagnostic information useful for debugging
        """
        # TODO: replace this by a real update (which depends on the action)
        flowsControllableByAgent = [10, 20, 30, 40]
        self.updateAllFlows(flowsControllableByAgent)

        self.updateViolations()
        self.updateObjective()
        self.updateFitness()

        self.stepNum += 1

        observation = self.x[:] # Make a copy of the state
        reward = self.fitness
        done = (self.stepNum == self.TOTAL_NUMBER_OF_EPISODES)
        info = None
        return observation, reward, done, info

    def reset(self):
        """ Reset the environment to its initial values """
        # Variables to be optimized
        self.x = [0.0] * 6

        # Fitness
        self.fitness = 0

        # Objective function
        self.objectiveScores = [0.0] * 6
        self.totalObjectiveScore = 0

        # Violations
        self.constraintValues = [0.0] * 9
        self.violations = 0
        self.violationPenalty = 0
        self.numViolations = 0
        self.violationsMultiplier = 100

        self.totalViolationsSum = 0

    def render(self, mode='human'):
        """ Renders the environment and the state of the flows """
        # Note, this may seem badly aligned but it is correct!
        # It becomes aligned after substituing the variables by
        # their real value
        content = """
                          x1 = {x1}
                        ------------> City
                       /                                            x5 = {x5}
                      /                 x2 = {x2}              ----------------> Farms
        Q1 = {Q1} --------------> Dam ----------------         /
                                                    |       /
                                                    |-------------------------> Lower triburary
                                                    |             x6 = {x6}
        Q2 = {Q2} ------------------------------------
                      \\                 x3 = {x3}
                       \\
                        -------------> Farms
                         x4 = {x2}
        """
        print(
            content.format(
                Q1=self.Q1,
                Q2=self.Q2,
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
