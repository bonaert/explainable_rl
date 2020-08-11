import dataclasses
from collections import deque
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
import argparse

from nicetypes import NumpyArray

ALWAYS = 2
FIRST_RUN = 1
NEVER = 0


def get_args():
    parser = argparse.ArgumentParser()

    # Main parameters
    parser.add_argument("--env-name", type=str, default="NotProvided")

    parser.add_argument("--num-training-episodes", type=int, default=50000)
    parser.add_argument("--eval-frequency", type=int, default=100)
    parser.add_argument("--render-rounds", type=int, default=1)
    parser.add_argument("--num-test-episodes", type=int, default=5)

    parser.add_argument("--subgoal-testing-frequency", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-buffer-size", type=int, default=2_000_000)

    parser.add_argument("--num-update-steps-when-training", type=int, default=40)

    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--all-levels-maximize-reward", action="store_true")
    parser.add_argument("--reward-present-in-input", action="store_true")

    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ignore-rewards-except-top-level", action="store_true")

    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--run-on-cluster", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--use-teacher", action="store_true")

    args = parser.parse_args()
    return args


class ActionRepeatEnvWrapper(object):
    def __init__(self, env, action_repeat=3, reward_scale=1):
        env.spec.reward_threshold *= reward_scale
        self._env = env
        self.reward_scale = reward_scale
        self.action_repeat = action_repeat

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        r = 0.0
        for _ in range(self.action_repeat):
            obs_, reward_, done_, info_ = self._env.step(action)
            r = r + reward_

            # SUPER IMPORTANT: they replace the -100 CRASH reward by 0
            # e.g. they don't penalize the crash
            if done_ and self.action_repeat != 1:
                return obs_, 0.0, done_, info_

            if self.action_repeat == 1:
                return obs_, self.reward_scale * r, done_, info_
        return obs_, self.reward_scale * r, done_, info_


def polyak_average(source_network: torch.nn.Module, target_network: torch.nn.Module, polyak_coeff: float):
    """ Given two networks with the same architecture, updates the parameters in the target network
    using the parameters of the source network, according to the formula:
        param_target := param_target * polyak_coeff + (1 - polyak_coeff) * param_source
    """
    with torch.no_grad():
        for param, param_target in zip(source_network.parameters(), target_network.parameters()):
            # We use the inplace operators to avoid creating a new tensor needlessly
            param_target.mul_(polyak_coeff)
            param_target.add_((1 - polyak_coeff) * param.data)


def get_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float()
    else:
        return torch.tensor(x, dtype=torch.float32)


def permissive_get_tensor(x: Optional[Union[np.ndarray, torch.Tensor]]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    elif torch.is_tensor(x):
        return x.float()
    else:
        return torch.tensor(x, dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, max_size: int, num_transition_dims: int):
        self.num_transition_dims = num_transition_dims
        self.buffers = [deque(maxlen=max_size) for _ in range(num_transition_dims)]

    def size(self) -> int:
        # All buffers have the same size, so we abitrarily examine the first one
        return len(self.buffers[0])

    def add(self, transition: tuple):
        assert len(transition) == self.num_transition_dims
        for i, element in enumerate(transition):
            self.buffers[i].append(element)

    def add_many(self, transitions: List[tuple]):
        for transition in transitions:
            self.add(transition)

    def get_batch(self, batch_size: int) -> List[torch.Tensor]:
        indices = np.random.randint(low=0, high=self.size(), size=batch_size)
        results = []
        for buffer in self.buffers:
            values_list = [buffer[index] for index in indices]
            # There are always values, except one case: the goal for the top level are None, and therefore can't be changed to into
            # a tensor of floats. In that case, they stay as a list of None's, and they will be ignore by the critic and actor
            if values_list[0] is not None:
                values_tensor = torch.as_tensor(values_list, dtype=torch.float32)
            else:
                values_tensor = values_list
            results.append(values_tensor)

        return results


def get_range_and_center(low: np.array, high: np.array) -> Tuple[np.ndarray, np.ndarray]:
    bounds_range = (high - low) / 2  # The whole range [low, high] has size (high - low) and the range is half of it
    center = (high + low) / 2  # The center of the interval [low, high] is the average of the extremes
    return bounds_range, center


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)

    raise TypeError('Not serializable')


def get_plan(agent, initial_state: NumpyArray, num_iters: int, goal_has_reward=False):
    # TODO(idea): instead of a fixed number of iterations, stop when the uncertainty gets too high
    # e.g. show only the steps the future steps the model is confident

    current_state = initial_state
    goals = []
    for i in range(num_iters):
        goal = agent.sample_actions(current_state, goal=None, deterministic=True, compute_log_prob=False)
        if goal_has_reward:
            goal = goal[:-1]  # Remove the desired reward from the goal (only keep the state)
        goals.append(goal)
        current_state = goal

    return goals
