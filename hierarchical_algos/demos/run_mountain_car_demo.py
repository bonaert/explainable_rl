import gym
import numpy as np
from pathlib import Path

import sys

folder_above = Path(__file__).resolve().parent.parent.as_posix()
teacher_folder = (Path(__file__).resolve().parent.parent / 'teacher').as_posix()
sys.path.append(folder_above)
sys.path.append(teacher_folder)

from hac_general import evaluate_hac, load_hac

print()
print("------------------------------------------------------------------------------------------------------------")
print("!! Pleasure ensure you have installed the gym_with_goal_visualisation, otherwise you won't see the goals !!")
print("!! The README.md in the gym_with_goal_visualisation explains how to install the environments !!")
print("------------------------------------------------------------------------------------------------------------")
print()
print()


if __name__ == '__main__':
    current_env = gym.make("MountainCarContinuous-v0")
    print("Action space: Low %s\tHigh %s" % (current_env.action_space.low, current_env.action_space.high))
    print("State space: Low %s\tHigh %s" % (current_env.observation_space.low, current_env.observation_space.high))

    pretrained_hac_dir = f"../../pretrained_hac/MountainCar_sac_2_hac_general_levels_h_20_v20"
    print(f"Pretrained HAC directory: {pretrained_hac_dir}")

    current_hac_params = load_hac(pretrained_hac_dir)
    num_successes, success_rate, reached_subgoal_rate, rewards, steps_per_episode = evaluate_hac(current_hac_params, current_env, render_rounds=100000, num_evals=100)

    print("\nSuccess rate (%d/%d): %.3f" % (num_successes, len(rewards), success_rate))
    # noinspection PyStringFormat
    print("Reward: %.3f +- %.3f" % (np.mean(rewards), np.std(rewards)))
    # noinspection PyStringFormat
    print("Number of steps: %.3f +- %.3f" % (np.mean(steps_per_episode), np.std(steps_per_episode)))
