import gym

from teacher.training.ddpg import ddpg_run_from_disk
# from teacher.training.sac import sac_run_from_disk
from teacher.training.sacEntropyAdjustment import sac_run_from_disk

if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')
    # ddpg_run_from_disk(env, has_scaler=True, render=True)
    sac_run_from_disk(env, has_scaler=True, render=True)
    env.close()  # To avoid benign but annoying errors when the gym render window closes
