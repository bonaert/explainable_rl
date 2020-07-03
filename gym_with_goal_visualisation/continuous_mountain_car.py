# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.seed()
        self.reset()

        self.carwidth = 40
        self.carheight = 20
        self.screen_width = 600
        self.screen_height = 400
        self.world_width = self.max_position - self.min_position
        self.scale = self.screen_width / self.world_width
        self.clearance = 10

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position)

        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    #    def get_state(self):
    #        return self.state

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def update_goal(self, subgoal, trans, arrow_line=None, arrow_triangle=None):
        pos, speed = subgoal
        # Update transformation
        screen_width = 600
        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        trans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        trans.set_rotation(math.cos(3 * pos))

        # Update the arrow size
        if arrow_line is not None and arrow_triangle is not None:
            self.update_arrow(speed, arrow_line, arrow_triangle)

    def update_arrow(self, speed, line, triangle):
        line.start = (0, self.carheight)
        line.end = (speed * 100 + sgn(speed) * 5, self.carheight)
        triangle.v[0] = (line.end[0], self.carheight + 5)  # Up
        triangle.v[1] = (line.end[0], self.carheight - 5)  # Down
        triangle.v[2] = (line.end[0] + sgn(speed) * 10, self.carheight)  # Extreme

    def create_trans_and_add_car(self, rendering, *color):
        l, r, t, b = -self.carwidth / 2, self.carwidth / 2, self.carheight, 0

        car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        car.set_color(*color)
        car.add_attr(rendering.Transform(translation=(0, self.clearance)))
        trans = rendering.Transform()
        car.add_attr(trans)
        return car, trans

    def create_arrow(self, rendering, car_trans, speed):
        line = rendering.Line((0, self.carheight / 2), (speed, self.carheight / 2))
        line.add_attr(car_trans)
        line.set_color(0.2, 0.2, 0.2)
        line.linewidth = 20

        arrow = rendering.FilledPolygon(
            [(line.end[0], self.carheight + 10),  # Up
             (line.end[0], self.carheight - 10),  # Down
             (line.end[0] + sgn(speed) * 10, self.carheight)]  # Extreme
        )
        arrow.add_attr(car_trans)
        arrow.set_color(0.2, 0.2, 0.2)

        return line, arrow

    def render(self, goal=None, end_goal=None, mode='human', plan_subgoals=None):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            ################ Car ################
            car0, self.cartrans0 = self.create_trans_and_add_car(rendering, 0, 0, 0)
            self.speed_car, self.speed_car_arrow = self.create_arrow(rendering, self.cartrans0, self.state[1])
            self.viewer.add_geom(car0)
            self.viewer.add_geom(self.speed_car)
            self.viewer.add_geom(self.speed_car_arrow)

            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans0)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans0)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            #####################################

            ################ Goal ################
            if goal is not None:
                car1, self.cartrans1 = self.create_trans_and_add_car(rendering, 1, 0.2, 0.2)
                self.speed_goal1, self.speed_goal1_arrow = self.create_arrow(rendering, self.cartrans1, goal[1])
                self.viewer.add_geom(car1)
                self.viewer.add_geom(self.speed_goal1)
                self.viewer.add_geom(self.speed_goal1_arrow)
                self.show_goal = True
            else:
                self.show_goal = False
            ######################################

            ############## End Goal ##############
            if end_goal is not None:
                car2, self.cartrans2 = self.create_trans_and_add_car(rendering, 0.2, 0.2, 1)
                self.speed_goal2, self.speed_goal2_arrow = self.create_arrow(rendering, self.cartrans2, end_goal[1])
                self.viewer.add_geom(car2)
                self.viewer.add_geom(self.speed_goal2)
                self.viewer.add_geom(self.speed_goal2_arrow)
                self.show_end_goal = True
            else:
                self.show_end_goal = False
            ######################################

            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        self.update_goal(self.state, self.cartrans0, self.speed_car, self.speed_car_arrow)

        if self.show_goal:
            self.update_goal(goal, self.cartrans1, self.speed_goal1, self.speed_goal1_arrow)

        if self.show_end_goal:
            self.update_goal(end_goal, self.cartrans2, self.speed_goal2, self.speed_goal2_arrow)

        if plan_subgoals is not None:
            for i, subgoal in enumerate(plan_subgoals):
                color = 1 - i / len(plan_subgoals)
                car_subgoal, cartrans_subgoal = self.create_trans_and_add_car(rendering, 0, color, 0)
                speed_subgoal, speed_subgoal_arrow = self.create_arrow(rendering, cartrans_subgoal, subgoal[1])
                self.update_goal(subgoal, cartrans_subgoal, speed_subgoal, speed_subgoal_arrow)
                self.viewer.add_onetime(car_subgoal)
                self.viewer.add_onetime(speed_subgoal)
                self.viewer.add_onetime(speed_subgoal_arrow)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
