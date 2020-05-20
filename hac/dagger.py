import copy
import random
from typing import Tuple, Callable, List

import gym
import torch
import numpy as np
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from common import ReplayBuffer
from sac import SacActor
from networks.simple import SacPolicy
from training.sac import get_policy_and_scaler

NumpyArray = np.ndarray
#                          State        Goal       Action
Level1Transition = Tuple[NumpyArray, NumpyArray, NumpyArray]


class Dagger:
    def __init__(self, expert: SacPolicy, level_2_policy: SacActor, level_1_policy: SacActor,
                 horizon_length: int, probability_use_level_1: float, env: gym.Env,
                 reached_goal_fn: Callable[[NumpyArray, NumpyArray], bool]):
        self.expert = expert
        self.level_2_policy = level_2_policy
        self.level_1_policy = level_1_policy
        self.horizon_length = int(horizon_length)
        self.probability_use_level_1 = probability_use_level_1
        self.reached_goal_fn = reached_goal_fn

        self.num_agents_taught = 20
        self.num_trajectories = 50
        self.env = env

    def teach_hrl_agent(self) -> Tuple[SacActor, SacActor]:
        current_agent_1 = self.level_1_policy
        current_agent_2 = self.level_2_policy
        replay_buffer_1 = ReplayBuffer(max_size=2_000_000, num_transition_dims=3)
        replay_buffer_2 = ReplayBuffer(max_size=2_000_000, num_transition_dims=2)
        for i in range(self.num_agents_taught):
            with torch.no_grad():
                new_experiences = []
                for _ in tqdm(range(self.num_trajectories)):
                    done = False
                    state = self.env.reset()

                    while not done:
                        if random.random() < self.probability_use_level_1:
                            goal, logprob = self.level_2_policy.sample_actions(state, goal=None, compute_log_prob=True)
                            end_state, done = self.rollout(state, goal)
                        else:
                            num_steps = random.randint(int(0.75 * self.horizon_length), self.horizon_length)
                            level_1_transitions, end_state, done = self.expert_rollout(state, num_steps)

                            new_experiences.append((state, end_state))
                            replay_buffer_1.add_many(level_1_transitions)

                        state = end_state

                replay_buffer_2.add_many(new_experiences)

            current_agent_1 = self.train_new_agent(replay_buffer_1, level=1)
            current_agent_2 = self.train_new_agent(replay_buffer_2, level=2)

            self.evaluate_agent(current_agent_1, current_agent_2, num_episodes_to_render=2)

        return current_agent_1, current_agent_2

    def train_new_agent(self, replay_buffer: ReplayBuffer, level: int) -> SacActor:
        assert level == 1 or level == 2
        new_agent = copy.deepcopy(self.level_2_policy if level == 2 else self.level_1_policy)

        batch_size = 32
        optimizer = Adam(new_agent.parameters())
        loss_fn = MSELoss()
        # Go through the data 4 times
        for i in range(replay_buffer.size() // batch_size * 4):
            if level == 2:
                states, desired_goals = replay_buffer.get_batch(batch_size)
                outputted_goals, _ = new_agent(states, goal=None)
                loss = loss_fn(outputted_goals, desired_goals)
            else:  # Level 1
                states, goals, desired_actions = replay_buffer.get_batch(batch_size)
                outputted_actions, _ = new_agent(states, goals)
                loss = loss_fn(outputted_actions, desired_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return new_agent

    def rollout(self, state: NumpyArray, goal: NumpyArray) -> Tuple[NumpyArray, bool]:
        done = False
        for i in range(self.horizon_length):
            action, logprob = self.level_1_policy.sample_actions(state, goal, compute_log_prob=True)
            next_state, reward, done, _ = self.env.step(action)

            state = next_state

            if self.reached_goal_fn(state, goal) or done:
                break

        return state, done

    def expert_rollout(self, state: NumpyArray, num_steps: int) -> Tuple[List[Level1Transition], NumpyArray, bool]:
        done = False
        transitions = []
        for i in range(num_steps):
            action = self.expert.get_actions(torch.tensor(state).float())
            next_state, reward, done, _ = self.env.step(action)

            transitions.append([state, action])

            state = next_state

            if done:
                break

        goal = state
        full_transitions = [(state, goal, action) for (state, action) in transitions]

        return full_transitions, state, done

    def evaluate_agent(self, current_agent_1: SacActor, current_agent_2: SacActor, num_episodes_to_render=0, num_episodes=10):
        rewards, duration, logprobs1, logprobs2 = [], [], [], []
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            goal = None
            should_pick_new_goal = True
            total_reward = 0.0
            num_steps_with_current_goal = 0
            num_steps_in_episode = 0
            while not done:
                if should_pick_new_goal:
                    goal, logprob2 = current_agent_2.sample_actions(state, goal=None, deterministic=True, compute_log_prob=True)
                    num_steps_with_current_goal = 0
                    logprobs2.append(logprob2)

                action, logprob1 = current_agent_1.sample_actions(state, goal, deterministic=True, compute_log_prob=True)

                next_state, reward, done, _ = self.env.step(action)
                if i < num_episodes_to_render:
                    self.env.render(goal=goal)

                total_reward += reward
                state = next_state
                num_steps_with_current_goal += 1
                num_steps_in_episode += 1

                logprobs1.append(logprob1)

                should_pick_new_goal = self.reached_goal_fn(state, goal) or num_steps_with_current_goal >= self.horizon_length

            rewards.append(total_reward)
            duration.append(num_steps_in_episode)

            print(f"Episode {i} - did {num_steps_in_episode} steps and collected reward {total_reward:.3f} ")

        print(f"Total rewards: {np.mean(rewards):.3f} +- {np.std(rewards):.3f}")
        print(f"Num steps: {np.mean(duration):.3f} +- {np.std(duration):.3f}")
        print(f"Log prob level 1: {np.mean(logprobs1):.3f} +- {np.std(logprobs1):.3f}")
        print(f"Log prob level 2: {np.mean(logprobs2):.3f} +- {np.std(logprobs2):.3f}")


if __name__ == '__main__':
    # Create environment
    current_env = gym.make("LunarLanderContinuous-v2")

    # Load teacher
    has_scaler = True
    sac_policy, scaler = get_policy_and_scaler(current_env, has_scaler)

    # Load student
    env_name = "LunarLanderContinuous-v2"
    use_sac = True
    num_levels = 2
    max_horizons = [10]  # This should stay at 20!
    # version = 9
    # current_directory = f"runs/{env_name}_{'sac' if use_sac else 'ddpg'}_{num_levels}_hac_general_levels_h_{'_'.join(map(str, max_horizons))}_v{version}"
    # current_directory = "runs/LunarLanderContinuous-v2_2_levels_h_10_40_v2"
    # hac_params = load_hac(current_directory)

    # Doing the distillation
    # TODO: incorporte scaler into policy so that I don't have to type this 20 times

    state_size = current_env.observation_space.shape[0]
    action_size = current_env.action_space.shape[0]
    action_low, action_high = current_env.action_space.low, current_env.action_space.high

    # Specific to lunar lander 2
    state_low = np.array([-2, -5, -3, -3, -10, -10, 0, 0], dtype=np.float32)
    state_high = np.array([2, 5, 3, 3, 20, 10, 1, 1], dtype=np.float32)
    state_distance_thresholds = np.array([0.04, 0.05, 0.5, 0.2, 1.0, 1, 0.5, 0.5])

    # state_low, state_high = env.observation_space.low, env.observation_space.high

    policy_level2 = SacActor(state_size, 0, state_size, action_low=state_low, action_high=state_high)
    policy_level1 = SacActor(state_size, state_size, action_size, action_low=action_low, action_high=action_high)
    horizon_length = 40

    dagger = Dagger(sac_policy, policy_level2, policy_level1,
                    horizon_length, probability_use_level_1=0.2, env=current_env,
                    reached_goal_fn=lambda state, goal:
                    (abs(state - goal) < state_distance_thresholds).all())
    new_level_1_policy, new_level_2_policy = dagger.teach_hrl_agent()

    dagger.evaluate_agent(new_level_1_policy, new_level_2_policy, num_episodes_to_render=100, num_episodes=100)

    # Saving the teached student model
    # save_hac(hac_params, directory="./dagger-taught/")
    # evaluate_hac(hac_params, env, render_rounds=1000, num_evals=1000)

    # TODO: implement the multiple predictions in advance things
    # TODO: integrate the expert in the goal learner in HAC


