import copy

import gym
from torch.optim import Adam

from networks.simple import DDPGPolicy, DDPGValueEstimator
from training.common import RunParams
from training.ddpg import DDPGParams, ddpg_train
from training.noise import OUNoise, NormalNoise

if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')

    run_params = RunParams(continuous_actions=True,
                           should_scale_states=True,
                           render_frequency=100,
                           entropy_coeff=0,
                           entropy_decay=1,
                           gamma=0.99,
                           use_tensorboard=True,
                           env_can_be_solved=False,
                           save_model_frequency=20)

    state_dim = 8
    action_dim = 2
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
        batch_size=100,
        polyak=0.999,
        noise_coeff=0.2,
        noise_source=OUNoise(action_dim, mu=0.4, decay=1),  #OUNoise(action_dim, decay=0.99995),
        num_random_action_steps=2000,
        num_test_episodes=10,
        test_frequency=50
    )

    ddpg_train(env, run_params, ddpg_params)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
