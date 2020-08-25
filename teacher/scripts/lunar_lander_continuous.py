import copy

import gym
from torch.optim import Adam

from teacher.networks.simple import DDPGPolicy, DDPGValueEstimator, SacPolicy, SacValueEstimator
from teacher.training.common import RunParams
from teacher.training.ddpg import DDPGParams, ddpg_train
from teacher.training.noise import OUNoise, NormalNoise
from teacher.training.sac import SacParams, sac_train
# from teacher.training.sacEntropyAdjustment import SacEntropyAdjustmentParams, sac_entropy_adjustment_train

if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')

    run_params = RunParams(continuous_actions=True,
                           should_scale_states=True,
                           render_frequency=0,
                           entropy_coeff=0,
                           entropy_decay=1,
                           gamma=0.99,
                           use_tensorboard=False,
                           env_can_be_solved=False,
                           save_model_frequency=20)

    state_dim = 8
    action_dim = 2
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
    #     batch_size=100,
    #     polyak=0.999,
    #     noise_coeff=0.2,
    #     noise_source=OUNoise(action_dim, mu=0.4, decay=1),  #OUNoise(action_dim, decay=0.99995),
    #     num_random_action_steps=2000,
    #     num_test_episodes=10,
    #     test_frequency=50
    # )
    #
    # ddpg_train(env, run_params, ddpg_params)
    sac_policy = SacPolicy(state_dim, action_dim, env.action_space.low, env.action_space.high)
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
        batch_size=100,
        polyak=0.995,
        num_random_action_steps=10000,
        alpha=0.2,
        num_test_episodes=10,
        test_frequency=20
    )

    # sac_params = SacEntropyAdjustmentParams(
    #     env=env,
    #     policy=sac_policy,
    #     policy_target=copy.deepcopy(sac_policy),
    #     value_estimator1=sac_value_estimator1,
    #     value_estimator2=sac_value_estimator2,
    #     value_estimator1_target=copy.deepcopy(sac_value_estimator1),
    #     value_estimator2_target=copy.deepcopy(sac_value_estimator2),
    #     policy_optimizer=Adam(sac_policy.parameters(), lr=1e-3),  # Same LR for both policy and value
    #     value_optimizer=Adam(value_parameters, lr=1e-3),
    #     replay_buffer_size=1000000,
    #     batch_size=100,
    #     polyak=0.995,
    #     num_random_action_steps=10000,
    #     num_test_episodes=10,
    #     test_frequency=20,
    #     initial_alpha=0.1,
    # )

    sac_train(env, run_params, sac_params)
    # sac_entropy_adjustment_train(env, run_params, sac_params)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
