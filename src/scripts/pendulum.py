import copy

import gym
import torch
from torch.optim import Adam

from networks.simple import SimplePolicyContinuous, SimplePolicyContinuous2, SimpleCritic2, DDPGPolicy, \
    DDPGValueEstimator, SacPolicy, SacValueEstimator
from src.training.reinforce import reinforceTraining
from training.actor_critic import actor_critic_train_per_step
from training.common import RunParams
from training.ddpg import DDPGParams, ddpg_train
from training.noise import OUNoise
from training.sac import SacParams, sac_train

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
                           should_scale_states=False,
                           render_frequency=20,
                           entropy_coeff=0,
                           entropy_decay=1,
                           gamma=0.99,
                           use_tensorboard=True,
                           env_can_be_solved=False,
                           save_model_frequency=20)

    # ddpg_policy = DDPGPolicy(3, 1, env.action_space.high, env.action_space.low)
    # ddpg_value_estimator = DDPGValueEstimator(3, 1)
    # ddpg_params = DDPGParams(
    #     policy=ddpg_policy,
    #     policy_target=copy.deepcopy(ddpg_policy),
    #     value_estimator=ddpg_value_estimator,
    #     value_estimator_target=copy.deepcopy(ddpg_value_estimator),
    #     policy_optimizer=Adam(ddpg_policy.parameters(), lr=1e-4),
    #     value_optimizer=Adam(ddpg_value_estimator.parameters(), lr=1e-3),
    #     replay_buffer_size=1000000,
    #     update_frequency=50,
    #     update_start=1000,
    #     batch_size=100,
    #     polyak=0.999,
    #     noise_coeff=0.1,
    #     noise_source=OUNoise(1),
    #     num_random_action_steps=10000,
    #     num_test_episodes=10
    # )
    #
    # ddpg_train(env, run_params, ddpg_params)

    sac_policy = SacPolicy(3, 1, env.action_space.high, env.action_space.low)
    sac_value_estimator1 = SacValueEstimator(3, 1)
    sac_value_estimator2 = SacValueEstimator(3, 1)
    value_parameters = list(sac_value_estimator1.parameters()) + list(sac_value_estimator2.parameters())

    sac_params = SacParams(
        policy=sac_policy,
        policy_target=copy.deepcopy(sac_policy),
        value_estimator1=sac_value_estimator1,
        value_estimator2=sac_value_estimator2,
        value_estimator1_target=copy.deepcopy(sac_value_estimator1),
        value_estimator2_target=copy.deepcopy(sac_value_estimator2),
        policy_optimizer=Adam(sac_policy.parameters(), lr=1e-3),  # Same LR for both policy and value
        value_optimizer=Adam(value_parameters, lr=1e-3),
        replay_buffer_size=1000000,
        update_frequency=50,
        update_start=1000,
        batch_size=100,
        polyak=0.995,
        num_random_action_steps=10000,
        alpha=0.2,
        num_test_episodes=10,
        test_frequency=20
    )
    sac_train(env, run_params, sac_params)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
