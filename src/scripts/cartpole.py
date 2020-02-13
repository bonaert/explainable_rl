import gym
import torch

from src.policy.linear import SimplePolicyDiscrete
from src.training.reinforce import reinforceTraining

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.seed(50)
    torch.manual_seed(50)
    simple_policy = SimplePolicyDiscrete(input_size=4, output_size=2)
    optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-2)

    reinforceTraining(simple_policy, env, optimizer, continuous_actions=False)