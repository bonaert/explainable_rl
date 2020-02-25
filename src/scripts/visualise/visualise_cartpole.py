import gym

from src.networks.simple import SimplePolicyDiscrete
from src.training.common import load_model, run_policy_repeatedly

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    simple_policy = load_model(SimplePolicyDiscrete(input_size=4, output_size=2), env, "policy.data")
    run_policy_repeatedly(env, simple_policy, render=True)
    env.close()
