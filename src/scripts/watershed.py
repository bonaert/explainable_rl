import copy

import gym
import gym_watershed
import torch
from torch.optim import Adam

from src.networks.simple import SimplePolicyContinuous, DDPGPolicy, DDPGValueEstimator, SacPolicy, SacValueEstimator
from src.training.reinforce import reinforceTraining
from training.common import RunParams
from training.ddpg import DDPGParams, ddpg_train
from training.noise import OUNoise
from training.sac import SacParams, sac_train

if __name__ == "__main__":
    # env = gym.make('watershed-v0')
    env = gym.make('watershed-v0', limited_scenarios=False, bizarre_actions=True)

    simple_policy = SimplePolicyContinuous(input_size=env.observation_space.shape[0],
                                           output_size=env.action_space.shape[0])
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
                           save_model_frequency=20,
                           stop_at_threshold=False)

    # ddpg_policy = DDPGPolicy(state_dim, action_dim, env.action_space.high, env.action_space.low)
    # ddpg_value_estimator = DDPGValueEstimator(state_dim, action_dim)
    # ddpg_params = DDPGParams(
    #     policy=ddpg_policy,
    #     policy_target=copy.deepcopy(ddpg_policy),
    #     value_estimator=ddpg_value_estimator,
    #     value_estimator_target=copy.deepcopy(ddpg_value_estimator),
    #     policy_optimizer=Adam(ddpg_policy.parameters(), lr=1e-4),
    #     value_optimizer=Adam(ddpg_value_estimator.parameters(), lr=1e-3),
    #     replay_buffer_size=1000000,
    #     update_frequency=50,
    #     update_start=2000,
    #     batch_size=128,
    #     polyak=0.99,
    #     noise_coeff=1,
    #     noise_source=OUNoise(action_dim),
    #     num_random_action_steps=2000,
    #     num_test_episodes=10
    # )
    #
    # ddpg_train(env, run_params, ddpg_params)

    # class RewardScalerWrapper(object):
    #     def __init__(self, env: gym.Env):
    #         self._env = env
    #
    #     def __getattr__(self, name):
    #         return getattr(self._env, name)
    #
    #     def step(self, action):
    #         obs, reward, done, info = self._env.step(action)
    #         return obs, reward / 1000, done, info
    # env = RewardScalerWrapper(env)

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
        replay_buffer_size=1_000_000,
        batch_size=100,
        polyak=0.995,
        num_random_action_steps=1000,
        alpha=0.2,
        num_test_episodes=10,
        test_frequency=20
    )
    sac_train(env, run_params, sac_params)

    env.close()
