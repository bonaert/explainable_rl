import gym

from networks.simple import DDPGPolicy, DDPGValueEstimator
from training.common import load_model, load_scaler
from training.ddpg import ddpg_run

if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ddpg_policy = load_model(DDPGPolicy(state_dim, action_dim, env.action_space.high, env.action_space.low), env, "policy_target.data")
    ddpg_value_estimator = load_model(DDPGValueEstimator(state_dim, action_dim), env, "value_estimator_target.data")
    scaler = load_scaler(env, "scaler.data")
    ddpg_run(env, ddpg_policy, scaler=scaler, render=True)
    env.close()  # To avoid benign but annoying errors when the gym render window closes
