from typing import List

import gym
import numpy as np
import torch

from hac_general import HacParams, mk_transition, evaluate_hac, save_hac, load_hac, build_input
from networks.simple import SacPolicy
from training.common import scale_state
from training.sac import get_policy_and_scaler


class Distillator:
    def __init__(self, teacher: SacPolicy, scaler, hac_params: HacParams, env: gym.Env):
        self.teacher = teacher
        self.scaler = scaler
        self.hac_params = hac_params
        self.env = env

    def distill_into_hierarchical(self):
        for i in range(500):
            state = self.env.reset()
            done = False
            states, actions, rewards, next_states, dones = [], [], [], [], []
            while not done:
                scaled_state = scale_state(self.scaler, state) if self.scaler else state
                action = self.teacher.get_actions(torch.tensor(scaled_state).float(), deterministic=False)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state

            self.distill_step(states, actions, rewards, next_states, dones)

    def distill_step(self, states: List[np.ndarray], actions: List[np.ndarray],
                     rewards: List[float], next_states: List[np.ndarray], dones: List[bool]):
        assert len(states) == len(actions)
        assert len(states) == len(rewards)
        assert len(states) == len(next_states)
        assert len(states) == len(dones)

        # Assumption: 2 levels, level 1 normal has horizon 20
        goal_1 = None
        horizon_level1 = 10
        total_reward_level_2 = 0.0
        total_reward_level_1 = 0.0
        for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            # Update the goal
            if i % horizon_level1 == 0:
                goal_1_index = min(horizon_level1 * i, len(states)) - 1
                reward_level_2 = sum(rewards[i * horizon_level1: goal_1_index])
                goal_state_1 = states[goal_1_index]
                goal_1 = np.hstack([goal_state_1, reward_level_2])

                current_input_2 = build_input(state, total_reward_level_2, 1, hac_params)
                total_reward_level_2 += reward_level_2
                next_input_2 = build_input(goal_state_1, total_reward_level_2, 1, hac_params)

                discount_level_2 = 0 if (i * horizon_level1 >= len(states)) else self.hac_params.discount
                self.hac_params.policies[1].add_to_buffer(
                    mk_transition(current_input_2, goal_1, None, None, next_input_2, reward_level_2, None, discount_level_2)
                )

            # Step 1) train agent 1
            discount_level_1 = 0 if done else self.hac_params.discount

            current_input_1 = build_input(state, total_reward_level_2, 1, hac_params)
            total_reward_level_1 += reward
            next_input_1 = build_input(next_state, total_reward_level_2, 1, hac_params)
            self.hac_params.policies[0].add_to_buffer(
                mk_transition(current_input_1, action, None, None, next_input_1, reward, goal_1, discount_level_1)
            )

        for agent in self.hac_params.policies:
            agent.learn(num_updates=10)


if __name__ == '__main__':
    # Create environment
    env = gym.make("LunarLanderContinuous-v2")

    # Load teacher
    has_scaler = True
    sac_policy, scaler = get_policy_and_scaler(env, has_scaler)

    # Load student
    env_name = "LunarLanderContinuous-v2"
    use_sac = True
    num_levels = 2
    max_horizons = [10]  # This should stay at 20!
    version = 9
    current_directory = f"runs/{env_name}_{'sac' if use_sac else 'ddpg'}_{num_levels}_hac_general_levels_h_{'_'.join(map(str, max_horizons))}_v{version}"
    hac_params = load_hac(current_directory)

    # Doing the distillation
    distillator = Distillator(sac_policy, scaler, hac_params, env)
    distillator.distill_into_hierarchical()

    # Saving the teached student model
    save_hac(hac_params, directory="./taught/")
    evaluate_hac(hac_params, env, render_rounds=1000, num_evals=1000)

