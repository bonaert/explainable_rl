import gym
import gym_watershed
import torch

from src.networks.simple import SimplePolicyContinuous
from src.training.reinforce import reinforceTraining
from training.common import RunParams

if __name__ == "__main__":
    env = gym.make('watershed-v0')

    simple_policy = SimplePolicyContinuous(input_size=env.observation_space.shape[0], output_size=env.action_space.shape[0])
    optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-4)

    # TODO: doesn't solve it, need to improve this
    # Possible problems: the environment changes a lot, need to see if Karl
    # resets the environment parameters at each step too (if not, the problem is MUCH MUCH easier)

    run_params = RunParams(continuous_actions=True,
                           should_scale_states=False,
                           render_frequency=0,
                           entropy_coeff=0.1,
                           entropy_decay=0.985)

    reinforceTraining(simple_policy, env, optimizer, run_params)
