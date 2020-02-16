import gym
import torch
from torch.optim.lr_scheduler import StepLR

from src.networks.simple import SimplePolicyContinuous, SimpleCriticContinuous, SimplePolicyContinuous2, \
    SimpleCriticContinuous2
from src.training.reinforce import reinforceTraining
from training.actor_critic import actor_critic_train_per_episode, actor_critic_train_per_step
from training.common import save_model

if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    simple_policy = SimplePolicyContinuous2(input_size=env.observation_space.shape[0],
                                            output_size=env.action_space.shape[0])
    simple_critic = SimpleCriticContinuous2(input_size=env.observation_space.shape[0])

    if False:
        optimizer = torch.optim.Adam(params=list(simple_policy.parameters()) + list(simple_critic.parameters()),
                                     lr=5e-5)
        reinforceTraining(simple_policy, env, optimizer, continuous_actions=True, scale_state=True,
                          train_with_batches=True)
    else:
        # actorCriticTraining(simple_policy, simple_critic, env, optimizer, continuous_actions=True)
        optimizer = torch.optim.Adam([
            {'params': simple_critic.parameters(), 'lr': 0.00056},
            {'params': simple_policy.parameters(), 'lr': 0.00001}],
            lr=0.00001)

        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

        # actor_critic_train_per_episode(simple_policy, simple_critic, env, optimizer, continuous_actions=True)
        actor_critic_train_per_step(simple_policy, simple_critic, env, optimizer,
                                    continuous_actions=True, scheduler=lr_scheduler)

        save_model(simple_policy, "simple_policy.data")
        save_model(simple_critic, "simple_critic.data")

    env.close()  # To avoid benign but annoying errors when the gym render window closes
