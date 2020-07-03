import gym_watershed
import gym
import pytest
import numpy as np


def test_canCreateEnvironment():
    env = gym.make('watershed-v0')


def test_canRender():
    env = gym.make('watershed-v0')
    env.reset()
    env.render()


def test_actionMustBeNumpyArray():
    with pytest.raises(Exception):
        env = gym.make('watershed-v0')
        env.reset()
        env.step("hello")


def test_canRun():
    env = gym.make('watershed-v0')
    env.reset()
    for i in range(env.spec.max_episode_steps):
        _, _, done, _ = env.step(
            env.action_space.sample()
        )
        if done:
            break


def test_stopsAfterTheRightAmountOfIterations():
    env = gym.make('watershed-v0')
    env.reset()
    for i in range(env.spec.max_episode_steps):
        _, _, done, _ = env.step(
            env.action_space.sample()
        )
    assert done


def test_canStateRespectsUpperBounds():
    env = gym.make('watershed-v0')
    env.reset()
    state, _, _, _ = env.step(
        env.action_space.high
    )
    state2, _, _, _ = env.step(
        env.action_space.high
    )
    assert np.allclose(state, state2)


def test_canStateRespectsLowerBounds():
    env = gym.make('watershed-v0')
    env.reset()
    state, _, _, _ = env.step(
        env.action_space.low
    )
    state2, _, _, _ = env.step(
        env.action_space.low
    )
    assert np.allclose(state, state2)
