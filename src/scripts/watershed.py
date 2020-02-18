import copy

import gym
import gym_watershed
import torch
from torch.optim import Adam

from src.networks.simple import SimplePolicyContinuous, DDPGPolicy, DDPGValueEstimator
from src.training.reinforce import reinforceTraining
from training.common import RunParams
from training.ddpg import DDPGParams, ddpg_train
from training.noise import OUNoise

if __name__ == "__main__":
    env = gym.make('watershed-v0')

    simple_policy = SimplePolicyContinuous(input_size=env.observation_space.shape[0], output_size=env.action_space.shape[0])
    optimizer = torch.optim.Adam(params=simple_policy.parameters(), lr=1e-4)

    # TODO: doesn't solve it, need to improve this
    # Possible problems: the environment changes a lot, need to see if Karl
    # resets the environment parameters at each step too (if not, the problem is MUCH MUCH easier)

    # run_params = RunParams(continuous_actions=True,
    #                        should_scale_states=False,
    #                        render_frequency=0,
    #                        entropy_coeff=0.1,
    #                        entropy_decay=0.985)
    #
    # reinforceTraining(simple_policy, env, optimizer, run_params)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    run_params = RunParams(continuous_actions=True,
                           should_scale_states=False,
                           render_frequency=0,
                           entropy_coeff=0,
                           entropy_decay=1,
                           gamma=0.99,
                           use_tensorboard=True,
                           env_can_be_solved=False,
                           save_model_frequency=10,
                           stop_at_threshold=False,
                           maximum_episodes=100)

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
        update_start=2000,
        batch_size=128,
        polyak=0.99,
        noise_coeff=1,
        noise_source=OUNoise(action_dim),
        num_random_action_steps=2000,
        num_test_episodes=10
    )

    ddpg_train(env, run_params, ddpg_params)

    env.close()
