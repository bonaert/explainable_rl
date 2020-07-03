import gym
import torch

from teacher.networks.simple import SimplePolicyDiscrete, SimplePolicyContinuous2, SimpleCritic2
from teacher.training.actor_critic import actor_critic_train_per_step
from teacher.training.common import RunParams

if __name__ == "__main__":
    env = gym.make('Acrobot-v1')

    run_params = RunParams(continuous_actions=False,
                           should_scale_states=True,
                           train_with_batches=True,
                           render_frequency=50,
                           entropy_coeff=0.005,
                           entropy_decay=0.99)

    # TODO: doesn't solve it, need to improve this

    # simple_policy = SimplePolicyDiscrete(input_size=6, output_size=3)
    # optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=5e-6)
    # reinforceTraining(simple_policy, env, optimizer, run_params)

    simple_policy = SimplePolicyDiscrete(input_size=6, output_size=3)
    simple_critic = SimpleCritic2(input_size=6)
    optimizer = torch.optim.Adam([
        {'params': simple_critic.parameters(), 'lr': 0.00056},
        {'params': simple_policy.parameters(), 'lr': 0.00001}],
        lr=0.00001)
    actor_critic_train_per_step(simple_policy, simple_critic, env, optimizer, run_params, lr_scheduler=None)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
