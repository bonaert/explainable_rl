from pathlib import Path

import gym
from gym import wrappers

from src.training.ddpg import ddpg_run_from_disk
from src.training.sac import sac_run_from_disk

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3')
    # env = gym.make('BipedalWalkerHardcore-v3')
    # ddpg_run_from_disk(env, has_scaler=True, render=True)
    # env = wrappers.Monitor(env, Path.cwd() / "BidepalWalkerHardcode-v3", force=True)
    sac_run_from_disk(env, has_scaler=True, render=True, run_once=False)
    env.close()  # To avoid benign but annoying errors when the gym render window closes
