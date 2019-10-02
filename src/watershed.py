import gym


# from gym import error, spaces, utils
# from gym.utils import seeding


# General comments:
# The Watershed problem seems to be a problem of continuous optimization instead of an RL
# problem. There isn't a clear set of actions and a clear definition of reward (we could
# define some, but it isn't explicit). The use of PSO as a technique to solve the problem
# seems to confirm that this isn't a RL problem.
# Idea: using a standard RL environment (ex. of the OpenAI gym environments) would
# be both simpler, faster and guarantees that the code is correct.
# TODO: talk about it


class WatershedEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
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
        self.Q1 = [0] * 100
        self.Q2 = [0] * 50

        # "S is storage capacity of the dam (in L^3)"
        self.S = [0] * 150

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
        # In his code, he receives a state X and updates his local variables
        # his method implicitly does self.setState(x)
        # I don't do that currently, because it's quite a weird behavior
        # TODO: check if this is going to be a problem

        for i in range(6):
            self.objectiveScores[i] = self.a[i] * self.x[i] ** 2 + self.b[i] * self.x[i] + self.c[i]

        self.totalObjectiveScore = sum(self.objectiveScores)

    def updateFitness(self):
        self.fitness = self.totalObjectiveScore - self.violationPenalty

    def setVariables(self, directVariables):
        # Direct variables = [x[0], x[1], x[3], x[5]]

        # First update the indirect variables
        # TODO: this is directly translated from the Java code, but it's very weird code.
        # In the first line, he uses the new values of the direct variables
        # to compute the indirect variables
        # In the second line, he uses a mix of the new values of the direct variables
        # and the old value of the indirect variable x3. I don't understand why he doesn't
        # use the new value of x3 (which we compute on the first line). This seems to
        # me that it will be incorrect.
        # Think about it to see if I understand, otherwise contact him.
        self.x[2] = self.Q2 - directVariables[2]  # x3 = Q2 - x4
        self.x[4] = directVariables[1] + self.x[2] - directVariables[3]  # x5 = x2 + x3 - x6

        # Then update the direct variables
        self.x[0] = directVariables[0]
        self.x[1] = directVariables[1]
        self.x[3] = directVariables[2]
        self.x[5] = directVariables[3]

    def step(self, action):
        """
        Returns observation, reward, done, info = env.step(action)
            - observation (object)
            - reward (float)
            - done (boolean)
            - info (dict): diagnostic information useful for debugging
        """
        self.updateViolations()
        self.updateObjective()
        self.updateFitness()

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
