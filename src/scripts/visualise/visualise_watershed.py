import gym
import gym_watershed
from training.sac import sac_run_from_disk

if __name__ == "__main__":
    env = gym.make('watershed-v0', limited_scenarios=False, bizarre_actions=True)

    # There might be a bug, because the average last reward is sometimes 185 +- 25
    # and other times it's 205 +- 25. This isn't normal at all.
    # TODO: find out why this strange behavior happens and if it's due to an error / bug, fix it!

    sac_run_from_disk(env, has_scaler=False, render=False, run_once=False)
    env.close()  # To avoid benign but annoying errors when the gym render window closes
