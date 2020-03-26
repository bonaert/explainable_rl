import numpy as np


def get_reward_function(dims):
    def hiro_controller_reward(z, subgoal, next_z, scale):
        z = z[:dims]
        next_z = next_z[:dims]
        reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
        return reward

    return hiro_controller_reward
