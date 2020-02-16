import gym
import torch

from networks.simple import SimplePolicyContinuous, SimplePolicyContinuous2, SimpleCritic2
from src.training.reinforce import reinforceTraining
from training.actor_critic import actor_critic_train_per_step
from training.common import RunParams

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    # simple_policy = SimplePolicyContinuous(input_size=3, output_size=1)
    # optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-3)
    # reinforceTraining(simple_policy, env, optimizer, continuous_actions=True)

    simple_policy = SimplePolicyContinuous2(input_size=3, output_size=1)
    simple_critic = SimpleCritic2(input_size=3)
    optimizer = torch.optim.Adam(params=list(simple_policy.parameters()) + list(simple_critic.parameters()), lr=1e-3)

    run_params = RunParams(continuous_actions=True,
                           should_scale_states=True,
                           train_with_batches=False,
                           render_frequency=10,
                           entropy_coeff=0.1,
                           entropy_decay=0.985)

    actor_critic_train_per_step(simple_policy, simple_critic, env, optimizer, run_params, lr_scheduler=None)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
