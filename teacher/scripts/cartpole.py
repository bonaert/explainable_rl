import gym
import torch

from teacher.networks.simple import SimplePolicyDiscrete
from teacher.training.reinforce import reinforceTraining
from teacher.training.common import RunParams

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    simple_policy = SimplePolicyDiscrete(input_size=4, output_size=2)
    optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-2)

    run_params = RunParams(continuous_actions=False,
                           should_scale_states=False,
                           train_with_batches=False,
                           render_frequency=0,
                           entropy_coeff=0.005,
                           entropy_decay=0.99,
                           use_tensorboard=True,
                           env_can_be_solved=False,
                           save_model_frequency=10)
    reinforceTraining(simple_policy, env, optimizer, run_params)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
