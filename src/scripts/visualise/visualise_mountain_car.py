import gym

from src.training.ddpg import ddpg_run_from_disk
from src.training.sac import sac_run_from_disk

if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')

    ddpg_run_from_disk(env, has_scaler=True, render=False)
    # sac_run_from_disk(env, has_scaler=True, render=True)
    env.close()  # To avoid benign but annoying errors when the gym render window closes
