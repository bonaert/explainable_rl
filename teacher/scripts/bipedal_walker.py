import copy

import gym
from torch.optim import Adam

from teacher.networks.simple import DDPGPolicy, DDPGValueEstimator, SacValueEstimator, SacPolicy
from teacher.training.common import RunParams
from teacher.training.ddpg import DDPGParams, ddpg_train
from teacher.training.noise import OUNoise, NormalNoise
from teacher.training.sac import SacParams, sac_train
import numpy as np

if __name__ == "__main__":
    # env = gym.make('BipedalWalkerHardcore-v3')
    env = gym.make('BipedalWalker-v3')
    reward_scale = 5.0

    ###################################################
    # The trick to solve Bipedal and Bipedal Hardcore #
    ###################################################
    # To solve these environments, running SAC or DDPG
    # alone doesn't seem to work. The agent doesn't explore
    # very well so it gets stuck in very bad local minima.
    # To solve this, the trick is not in the algorithm
    # itself, but in the actions that we take.
    #
    # The Wrapper around the environment below changes the
    # step() function so that the given action is repeated
    # 3 times. It also does some reward scaling (I don't
    # think that matters). Originally it also added noise
    # to the observations and to the actions, but I removed
    # that and was still able to solve the BipedalWalker
    # and the BipedalWalkerHardcore environemnts. I think
    # the key is repeating the same action 3 times, because
    # that improves exploration considerably.
    #
    # Before trying this, I implemented the SAC algorithm
    # but that turned out not to be necessary, because
    # DDPG can solve the BipedalWalker environment if
    # I use the environment wrapper. I haven't tested
    # DDPG on the wrapped BipedalWalkerHardcore, but I'm
    # confident it will work too.
    #
    # Again, RL proves a problem where tricks matters a lot:
    # 1) In mountain car, the key is good exploration, which
    #    is done by adding the right kind of noise
    #    (OU instead of Gaussian)
    # 2) In Bipedal Walker, the key is again good exploration
    #    To achieve that, the key is to repeat the action 3 times
    #    No special noise is needed, Gaussian works fine
    #
    # Credit: the Wrapper below is a simplification of the
    #         environment wrapper present here:
    # https://github.com/createamind/DRL/blob/master/spinup/algos/sac1/sac1_BipedalWalker-v2_200ep.py


    class Wrapper(object):
        def __init__(self, env, action_repeat=3):
            self._env = env
            env.spec.reward_threshold *= reward_scale
            self.action_repeat = action_repeat

        def __getattr__(self, name):
            return getattr(self._env, name)

        def reset(self):
            obs = self._env.reset()
            return obs

        def step(self, action):
            r = 0.0
            for _ in range(self.action_repeat):
                obs_, reward_, done_, info_ = self._env.step(action)
                r = r + reward_
                if done_ and self.action_repeat != 1:
                    return obs_, 0.0, done_, info_
                if self.action_repeat == 1:
                    return obs_, r, done_, info_
            return obs_, reward_scale * r, done_, info_


    env = Wrapper(env, 3)

    run_params = RunParams(continuous_actions=True,
                           should_scale_states=True,
                           render_frequency=100,
                           entropy_coeff=0,
                           entropy_decay=1,
                           gamma=0.99,
                           use_tensorboard=True,
                           env_can_be_solved=False,
                           save_model_frequency=10)

    state_dim = 24
    action_dim = 4

    # ddpg_policy = DDPGPolicy(state_dim, action_dim, env.action_space.high, env.action_space.low)
    # ddpg_value_estimator = DDPGValueEstimator(state_dim, action_dim)
    # ddpg_params = DDPGParams(
    #     policy=ddpg_policy,
    #     policy_target=copy.deepcopy(ddpg_policy),
    #     value_estimator=ddpg_value_estimator,
    #     value_estimator_target=copy.deepcopy(ddpg_value_estimator),
    #     policy_optimizer=Adam(ddpg_policy.parameters(), lr=1e-4),
    #     value_optimizer=Adam(ddpg_value_estimator.parameters(), lr=1e-3),
    #     replay_buffer_size=2_000_000,
    #     update_frequency=50,
    #     update_start=1000,
    #     batch_size=128,
    #     polyak=0.995,
    #     noise_coeff=1,
    #     noise_source=NormalNoise(action_dim, decay=1),
    #     num_random_action_steps=10000,
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
        policy_optimizer=Adam(sac_policy.parameters(), lr=5e-4),  # Same LR for both policy and value
        value_optimizer=Adam(value_parameters, lr=5e-4),
        replay_buffer_size=2_000_000,
        batch_size=100,
        polyak=0.995,
        num_random_action_steps=10000,
        alpha=0.2,
        num_test_episodes=10,
        test_frequency=50
    )

    sac_train(env, run_params, sac_params)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
