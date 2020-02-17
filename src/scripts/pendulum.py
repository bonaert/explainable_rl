import copy

import gym
import torch
from torch.optim import Adam

from networks.simple import SimplePolicyContinuous, SimplePolicyContinuous2, SimpleCritic2, DDPGPolicy, \
    DDPGValueEstimator
from src.training.reinforce import reinforceTraining
from training.actor_critic import actor_critic_train_per_step
from training.common import RunParams
from training.ddpg import DDPGParams, ddpg_train

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    # simple_policy = SimplePolicyContinuous(input_size=3, output_size=1)
    # optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-3)
    # reinforceTraining(simple_policy, env, optimizer, continuous_actions=True)

    # TODO: doesn't solve it, need to improve this

    # simple_policy = SimplePolicyContinuous2(input_size=3, output_size=1)
    # simple_critic = SimpleCritic2(input_size=3)
    # optimizer = torch.optim.Adam(params=list(simple_policy.parameters()) + list(simple_critic.parameters()), lr=1e-3)
    #
    # run_params = RunParams(continuous_actions=True,
    #                        should_scale_states=True,
    #                        train_with_batches=False,
    #                        render_frequency=10,
    #                        entropy_coeff=0.1,
    #                        entropy_decay=0.985)

    # actor_critic_train_per_step(simple_policy, simple_critic, env, optimizer, run_params, lr_scheduler=None)

    run_params = RunParams(continuous_actions=True,
                           should_scale_states=True,
                           render_frequency=100,
                           entropy_coeff=0,
                           entropy_decay=1,
                           use_tensorboard=True,
                           env_can_be_solved=False)

    ddpg_policy = DDPGPolicy(3, 1)
    ddpg_value_estimator = DDPGValueEstimator(3, 1)
    ddpg_params = DDPGParams(
        policy=ddpg_policy,
        policy_target=copy.deepcopy(ddpg_policy),
        value_estimator=ddpg_value_estimator,
        value_estimator_target=copy.deepcopy(ddpg_value_estimator),
        policy_optimizer=Adam(ddpg_policy.parameters(), lr=1e-4),
        value_optimizer=Adam(ddpg_value_estimator.parameters(), lr=1e-3),
        replay_buffer_size=100000,
        update_frequency=50,
        update_start=1000,
        batch_size=128,
        polyak=0.995,
        noise_coeff=0.1,
        start_steps=10000
    )

    ddpg_train(env, run_params, ddpg_params)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
