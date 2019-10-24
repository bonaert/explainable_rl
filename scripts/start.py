import gym_watershed
import gym
env = gym.make('watershed-v0')

state, _, _, _ = env.step(
    env.action_space.low
)