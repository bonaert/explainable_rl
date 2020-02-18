import gym

from networks.simple import SimplePolicyDiscrete
from training.common import load_model, run_model_repeatedly

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    simple_policy = load_model(SimplePolicyDiscrete(input_size=4, output_size=2), env, "policy.data")
    run_model_repeatedly(simple_policy, env, continuous_actions=False, render=True)
    env.close()
