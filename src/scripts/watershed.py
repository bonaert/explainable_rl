import gym
import gym_watershed
import torch

from src.policy.linear import SimplePolicyContinuous
from src.training.reinforce import reinforceTraining

if __name__ == "__main__":
    env = gym.make('watershed-v0')
    env.seed(50)
    torch.manual_seed(50)
    simple_policy = SimplePolicyContinuous(input_size=env.observation_space.shape[0], output_size=env.action_space.shape[0])
    optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-2)

    reinforceTraining(simple_policy, env, optimizer, continuous_actions=True)