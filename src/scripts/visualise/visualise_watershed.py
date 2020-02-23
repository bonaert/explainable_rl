import gym
import gym_watershed
from training.sac import sac_run_from_disk

if __name__ == "__main__":
    env = gym.make('watershed-v0', limited_scenarios=False, bizarre_actions=True)
    sac_run_from_disk(env, has_scaler=False, render=False, run_once=False)
    env.close()  # To avoid benign but annoying errors when the gym render window closes
