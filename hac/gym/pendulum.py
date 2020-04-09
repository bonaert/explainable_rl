import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt

        newth = angle_normalize(th + newthdot * dt)  #####

        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}
        # return self.state, -costs, False, {}  #####

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()
        # return self.state  #####

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def create_arrow(self, rendering, trans, speed):
        line = rendering.Line((1, 0), (1, speed * 0.1))
        line.add_attr(trans)
        line.set_color(0.2, 0.2, 0.2)
        line.linewidth = 20

        arrow = rendering.FilledPolygon(
            [(1.05, line.end[1]),  # Up
             (0.95, line.end[1]),  # Down
             (1.0, line.end[1] + sgn(speed) * 0.05)]  # sgn(speed) * 10)]  # Extreme
        )
        arrow.add_attr(trans)
        arrow.set_color(0.2, 0.2, 0.2)

        return line, arrow

    def create_rod(self, rendering, length, radius, *color):
        rod = rendering.make_goal_circ(length, radius)
        rod.set_color(*color)
        pole_transform = rendering.Transform()
        rod.add_attr(pole_transform)
        return rod, pole_transform

    def update_arrow(self, line, arrow, speed):
        line.start = (1, 0)
        line.end = (1, speed * 0.1)
        arrow.v[0] = (1.05, line.end[1])  # Up
        arrow.v[1] = (0.95, line.end[1])  # Down
        arrow.v[2] = (1.00, line.end[1] + sgn(speed) * 0.05)  # Extreme

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_goal(self, goal, end_goal, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            ################ goal ################
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.line0, self.arrow0 = self.create_arrow(rendering, self.pole_transform, self.state[1])

            self.viewer.add_geom(rod)
            self.viewer.add_geom(self.line0)
            self.viewer.add_geom(self.arrow0)
            ######################################

            ################ goal ################
            rod1, self.pole_transform1 = self.create_rod(rendering, 1, .1, .8, .8, .3)
            self.line1, self.arrow1 = self.create_arrow(rendering, self.pole_transform1, goal[1])

            self.viewer.add_geom(rod1)
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.arrow1)
            ######################################

            ############## End Goal ##############
            rod2, self.pole_transform2 = self.create_rod(rendering, 1, .1, .3, .3, .8)
            self.viewer.add_geom(rod2)
            ######################################

            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        self.pole_transform1.set_rotation(goal[0] + np.pi / 2)
        self.pole_transform2.set_rotation(end_goal[0] + np.pi / 2)

        self.update_arrow(self.line0, self.arrow0, self.state[1])
        self.update_arrow(self.line1, self.arrow1, goal[1])

        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_goal_2(self, goal1, goal2, end_goal, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            ################ goal 1 ################
            rod1 = rendering.make_goal_circ(1, .1)
            rod1.set_color(.8, .8, .3)
            self.pole_transform1 = rendering.Transform()
            rod1.add_attr(self.pole_transform1)
            self.viewer.add_geom(rod1)
            ########################################

            ################ goal 2 ################
            rod2 = rendering.make_goal_circ(1, .1)
            rod2.set_color(.3, .8, .3)
            self.pole_transform2 = rendering.Transform()
            rod2.add_attr(self.pole_transform2)
            self.viewer.add_geom(rod2)
            ########################################

            ############### End Goal ###############
            rod3 = rendering.make_goal_circ(1, .1)
            rod3.set_color(.3, .3, .8)
            self.pole_transform3 = rendering.Transform()
            rod3.add_attr(self.pole_transform3)
            self.viewer.add_geom(rod3)
            ########################################

            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        #     self.viewer.add_onetime(self.img)

        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)

        self.pole_transform1.set_rotation(goal1[0] + np.pi / 2)

        self.pole_transform2.set_rotation(goal2[0] + np.pi / 2)

        self.pole_transform3.set_rotation(end_goal[0] + np.pi / 2)

        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)