import gym
import numpy as np
import sys
from pathlib import Path

folder_above = Path(__file__).resolve().parent.parent.as_posix()
main_folder = Path(__file__).resolve().parent.parent.parent.as_posix()
sys.path.append(folder_above)
sys.path.append(main_folder)

from hac_general import HacParams, train
from teacher.training.sac import get_policy_and_scaler

env = gym.make('LunarLanderContinuous-v2')

# Action space: Low [-1. -1.]	High [1. 1.]
# State space:  Low [-inf] x 8         High [inf] x 8
# State: x, y, vel.x, vel.y, lander.angle, angular_velocity, bool(left left on ground), bool(right leg on ground)
overriden_state_space_low = np.array([-2, -5, -3, -3, -5, -5, 0, 0], dtype=np.float32)
overriden_state_space_high = np.array([2,  5,  3,  3,  5,  5, 1, 1], dtype=np.float32)
state_distance_thresholds = [[0.2, 0.1, 0.2, 0.1, 0.3, 0.5, 1.0, 1.0]]

# Use pre-trained teachers we provide for the Lunar Lander environment
teacher, scaler = get_policy_and_scaler(env, has_scaler=True)
probability_to_use_teacher = 0.5

# Q bounds for the critic, to help learning
q_bound_low_list = [-40, -1000.0]
q_bound_high_list = [0.0, 300.0]

hac_params = HacParams(
    action_low=env.action_space.low,
    action_high=env.action_space.high,
    state_low=overriden_state_space_low,
    state_high=overriden_state_space_high,
    reward_low=[None, -1000],
    reward_high=[None, 200],
    batch_size=128,
    num_training_episodes=5000,
    num_levels=2,
    max_horizons=[40],
    discount=0.98,
    replay_buffer_size=2_000_000,
    subgoal_testing_frequency=0.1,
    state_distance_thresholds=state_distance_thresholds,
    num_update_steps_when_training=40,
    evaluation_frequency=100,
    save_frequency=100,
    env_threshold=200.0,
    env_name=env.spec.id,
    use_priority_replay=False,
    penalty_subgoal_reachability=-1000.0,
    use_sac=True,
    all_levels_maximize_reward=False,
    reward_present_in_input=False,
    num_test_episodes=10,
    learning_rates=[3e-4, 3e-4],

    # Teacher suff
    teacher=teacher,
    state_scaler=scaler,
    probability_to_use_teacher=probability_to_use_teacher,
    learn_low_level_transitions_from_teacher=True,

    # Logging
    use_tensorboard=False,

    # Q-bounds
    q_bound_low_list=q_bound_low_list,
    q_bound_high_list=q_bound_high_list
)

train(hac_params, env, render_rounds=2, directory="runs/")
