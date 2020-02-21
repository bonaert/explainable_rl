import copy

import gym
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.networks.simple import SimplePolicyContinuous, SimpleCritic, SimplePolicyContinuous2, \
    SimpleCritic2, DDPGPolicy, DDPGValueEstimator, SacPolicy, SacValueEstimator
from src.training.reinforce import reinforceTraining
from training.actor_critic import actor_critic_train_per_episode, actor_critic_train_per_step
from training.common import save_model, RunParams
from training.ddpg import DDPGParams, ddpg_train
from training.noise import OUNoise
from training.sac import SacParams, sac_train

if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    simple_policy = SimplePolicyContinuous2(input_size=state_dim, output_size=action_dim)
    simple_critic = SimpleCritic2(input_size=state_dim)

    run_params = RunParams(continuous_actions=True,
                           should_scale_states=True,
                           render_frequency=10,
                           entropy_coeff=0,
                           entropy_decay=1,
                           gamma=0.99,
                           use_tensorboard=True,
                           env_can_be_solved=True,
                           save_model_frequency=10,
                           stop_at_threshold=False,
                           maximum_episodes=100)

    if False:
        optimizer = torch.optim.Adam(params=list(simple_policy.parameters()) + list(simple_critic.parameters()),
                                     lr=5e-5)
        run_params = RunParams(continuous_actions=True,
                               should_scale_states=True,
                               render_frequency=0,
                               entropy_coeff=0.1,
                               entropy_decay=0.985)
        reinforceTraining(simple_policy, env, optimizer, run_params)
    elif False:
        optimizer = torch.optim.Adam([
            {'params': simple_critic.parameters(), 'lr': 0.00056},
            {'params': simple_policy.parameters(), 'lr': 0.00001}],
            lr=0.00001)

        # lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        lr_scheduler = None

        run_params = RunParams(continuous_actions=True,
                               should_scale_states=True,
                               train_with_batches=False,
                               render_frequency=0,
                               entropy_coeff=0.1,
                               entropy_decay=0.985,
                               use_tensorboard=True)

        # actor_critic_train_per_episode(simple_policy, simple_critic, env, optimizer, run_params, lr_scheduler)
        actor_critic_train_per_step(simple_policy, simple_critic, env, optimizer, run_params, lr_scheduler)

        save_model(simple_policy, env, "policy.data")
        save_model(simple_critic, env, "critic.data")
    elif True:
        ddpg_policy = DDPGPolicy(state_dim, action_dim, env.action_space.high, env.action_space.low)
        ddpg_value_estimator = DDPGValueEstimator(state_dim, action_dim)
        ddpg_params = DDPGParams(
            policy=ddpg_policy,
            policy_target=copy.deepcopy(ddpg_policy),
            value_estimator=ddpg_value_estimator,
            value_estimator_target=copy.deepcopy(ddpg_value_estimator),
            policy_optimizer=Adam(ddpg_policy.parameters(), lr=1e-4),
            value_optimizer=Adam(ddpg_value_estimator.parameters(), lr=1e-3),
            replay_buffer_size=1000000,
            update_frequency=50,
            update_start=1000,
            batch_size=128,
            polyak=0.9,
            noise_coeff=1,
            noise_source=OUNoise(action_dim),  # This noise is super important! Without it we can solve it. It feels
                                               # a bit like cheating though, because it might simply be overfitting to
                                               # the problem. It's able to solve mountain car though
            num_random_action_steps=0,
            num_test_episodes=10
        )

        ddpg_train(env, run_params, ddpg_params)
    elif False:
        sac_policy = SacPolicy(state_dim, action_dim, env.action_space.high, env.action_space.low)
        sac_value_estimator1 = SacValueEstimator(state_dim, action_dim)
        sac_value_estimator2 = SacValueEstimator(state_dim, action_dim)
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
            alpha=1,
            num_test_episodes=10,
            test_frequency=20
        )
        sac_train(env, run_params, sac_params)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
