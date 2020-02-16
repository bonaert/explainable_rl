import gym
import torch

from src.networks.simple import SimplePolicyContinuous, SimpleCriticContinuous
from src.training.reinforce import reinforceTraining
from training.actor_critic import actorCriticTraining

if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    # env.seed(50)
    # torch.manual_seed(50)
    simple_policy = SimplePolicyContinuous(input_size=env.observation_space.shape[0],
                                           output_size=env.action_space.shape[0])
    simple_critic = SimpleCriticContinuous(input_size=env.observation_space.shape[0])

    optimizer = torch.optim.Adam(params=list(simple_policy.parameters()) + list(simple_critic.parameters()), lr=5e-5)

    #reinforceTraining(simple_policy, env, optimizer, continuous_actions=True, scale_state=True, train_with_batches=True)
    actorCriticTraining(simple_policy, simple_critic, env, optimizer, continuous_actions=True)

    env.close()  # To avoid benign but annoying errors when the gym render window closes
