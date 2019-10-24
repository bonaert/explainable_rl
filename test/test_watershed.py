import gym_watershed
import gym
import pytest
import numpy as np


def test_canCreateEnvironment():
    env = gym.make('watershed-v0')


def test_canRender():
    env = gym.make('watershed-v0')
    env.render()


def test_actionMustBeNumpyArray():
    with pytest.raises(Exception):
        env = gym.make('watershed-v0')
        env.step("hello")


def test_actionMustBeHaveCorrectSize():
    with pytest.raises(Exception):
        env = gym.make('watershed-v0')
        env.step(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Too big
        )

    with pytest.raises(Exception):
        env = gym.make('watershed-v0')
        env.step(
            np.array([1, 2])  # Too small
        )


def test_actionMustBeWithinBounds():
    with pytest.raises(Exception):
        env = gym.make('watershed-v0')
        env.step(
            np.array([-10000, -10000, 10000, 1000])
        )


def test_canRun():
    env = gym.make('watershed-v0')
    done = False
    while not done:
        _, _, done, _ = env.step(
            env.action_space.sample()
        )


def test_canStateRespectsUpperBounds():
    env = gym.make('watershed-v0')
    state, _, _, _ = env.step(
        env.action_space.high
    )
    state2, _, _, _ = env.step(
        env.action_space.high
    )
    assert np.allclose(state, state2)


def test_canStateRespectsLowerBounds():
    env = gym.make('watershed-v0')
    state, _, _, _ = env.step(
        env.action_space.low
    )
    state2, _, _, _ = env.step(
        env.action_space.low
    )
    assert np.allclose(state, state2)
