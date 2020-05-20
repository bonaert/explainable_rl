from time import time
from typing import List

import gym
import numpy as np
import torch
from tqdm import tqdm

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
        for i in tqdm(range(500)):
            start = time()
            with torch.no_grad():
                actions, dones, next_states, rewards, states = self.collect_episode()
            print(sum(rewards))
            middle = time()
            self.distill_step(states, actions, rewards, next_states, dones)
            end = time()

            print()
            action2 = self.hac_params.policies[1].sample_action(states[0], None)
            base_action2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 230.0])
            print(f"Q-values level 2 (base action): {self.hac_params.policies[1].critic1(states[0], None, base_action2).item()}")
            print(f"Q-values level 2 (picked action): {self.hac_params.policies[1].critic1(states[0], None, action2).item()}")
            print(f"Picked action level 2: {action2}")

            goal = self.hac_params.policies[1].sample_action(states[0], None)
            input_1 = build_input(states[0], 0.0, 0, self.hac_params)
            action1 = self.hac_params.policies[0].sample_action(input_1, goal)
            base_action1 = np.array([0., 0.])
            print(f"Q-values level 1 (base action): {self.hac_params.policies[0].critic1(input_1, goal, base_action1).item()}")
            print(f"Q-values level 1 (picked action): {self.hac_params.policies[0].critic1(input_1, goal, action1).item()}")
            print(f"Picked action level 1: {action1}")

            # print(f"Collect experience {middle - start:.4f}\tAdding transitions to buffer and training the hierarchy {end - middle:.4f}")

            if i % 20 == 0:
                evaluate_hac(self.hac_params, self.env, render_rounds=1, num_evals=10)
                print()

    def collect_episode(self):
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
        return actions, dones, next_states, rewards, states

    def distill_step(self, states: List[np.ndarray], actions: List[np.ndarray],
                     rewards: List[float], next_states: List[np.ndarray], dones: List[bool]):
        assert len(states) == len(actions)
        assert len(states) == len(rewards)
        assert len(states) == len(next_states)
        assert len(states) == len(dones)

        # Assumption: 2 levels, level 1 normal has horizon 20
        goal_for_level1 = None
        horizon_level1 = 10
        total_reward_level_2 = 0.0
        total_reward_level_1 = 0.0
        for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            # Update the goal
            if i % horizon_level1 == 0:
                end_state_index = min(i + horizon_level1, len(states)) - 1
                state_for_goal_for_level1 = next_states[end_state_index]
                reward_level_2 = sum(rewards[i: end_state_index + 1])
                goal_for_level1 = np.hstack([state_for_goal_for_level1, reward_level_2])

                current_input_2 = build_input(state, total_reward_level_2, 1, hac_params)
                total_reward_level_2 += reward_level_2
                next_input_2 = build_input(state_for_goal_for_level1, total_reward_level_2, 1, hac_params)

                discount_level_2 = 0 if (i + horizon_level1 >= len(states)) else self.hac_params.discount
                # print(f"Level 2 - reward: {reward_level_2}\t discount: {discount_level_2}")
                self.hac_params.policies[1].add_to_buffer(
                    mk_transition(current_input_2, goal_for_level1, None, None, next_input_2, reward_level_2, None, discount_level_2)
                )

                # Every time level 2 picks a new goal and then lets level 1 act, then we should reset the counters
                # held by level 1. In our case, the only counter is the total reward it has collected during the episode,
                # which should be reset to 0
                total_reward_level_1 = 0.0

            # Step 1) train agent 1
            current_input_1 = build_input(state, total_reward_level_1, 0, hac_params)

            done_level_1 = (i + 1) % horizon_level1 == 0 or i == len(states) - 1
            discount_level_1 = 0 if done_level_1 else hac_params.discount
            reward_level_1 = 0 if done_level_1 else -1  # Here we're guaranteed to hit the goal, so we don't have to check that
            total_reward_level_1 += reward
            next_input_1 = build_input(next_state, total_reward_level_1, 0, hac_params)
            self.hac_params.policies[0].add_to_buffer(
                mk_transition(current_input_1, action, None, None, next_input_1, reward_level_1, goal_for_level1, discount_level_1)
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
    version = 11
    current_directory = f"runs/{env_name}_{'sac' if use_sac else 'ddpg'}_{num_levels}_hac_general_levels_h_{'_'.join(map(str, max_horizons))}_v{version}"
    hac_params = load_hac(current_directory)

    # Doing the distillation
    distillator = Distillator(sac_policy, scaler, hac_params, env)
    distillator.distill_into_hierarchical()

    # Saving the teached student model
    save_hac(hac_params, directory="./taught/")
    evaluate_hac(hac_params, env, render_rounds=1000, num_evals=1000)

    # This doesn't work, and it's probably because of the behavior cloning problem
    # I am implementing Dagger as an alternative

