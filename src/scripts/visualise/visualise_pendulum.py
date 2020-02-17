import gym

from networks.simple import DDPGPolicy, DDPGValueEstimator
from training.common import load_model
from training.ddpg import ddpg_run

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    ddpg_policy = load_model(DDPGPolicy(3, 1, env.action_space.high, env.action_space.low), env, "policy_target.data")
    ddpg_value_estimator = load_model(DDPGValueEstimator(3, 1), env, "value_estimator_target.data")
    ddpg_run(env, ddpg_policy, render=True)
    env.close()  # To avoid benign but annoying errors when the gym render window closes
