from gym_watershed import WatershedEnv
import gym

def test_canCreateEnvironment():
    env = gym.make('watershed-v0')

def test_canRender():
    env = gym.make('watershed-v0')
    env.render()

def test_canRun():
    env = gym.make('watershed-v0')
    done = False
    while not done:
        observation, reward, done, info = env.step(0)