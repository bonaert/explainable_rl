import argparse
import os
import random

import numpy as np
import gym
import torch

from hac import load_hac, train, evaluate_hac, HacParams
from common import ALWAYS, FIRST_RUN, NEVER, get_args

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="NotProvided")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-on-cluster", action="store_true")
    args = parser.parse_args()

    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    # noinspection PyUnreachableCode
    ##################################
    #     Environment parameters     #
    ##################################
    if args.env_name == "NotProvided":
        # env_name = "AntMaze"
        # env_name = "MountainCarContinuous-v0"
        # env_name = "Pendulum"
        env_name = "LunarLanderContinuous-v2"
    else:
        env_name = args.env_name

    if env_name == "AntMaze":
        # distance_thresholds = [0.1, 0.1]  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L106
        # max_horizons = 10  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L27
        # action_noise_coeffs = np.array([0.1] * current_action_size),  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L131
        # subgoal_noise_coeffs = np.array([0.03] * current_state_size),  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L132
        raise Exception("TODO")
    elif env_name == "MountainCarContinuous-v0":
        current_env = gym.make("MountainCarContinuous-v0")
        current_goal_state = np.array([0.48, 0.04])  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L45

        # noinspection PyUnreachableCode
        if True:
            num_levels = 2
            max_horizons = [20, 20]  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L50
            # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L46
            distance_thresholds = [[0.01, 0.02],    # We want to have precise subgoals
                                   [0.1, 10.0]]   # But for the goal I only care about the position (not the speed)
        else:
            num_levels = 3
            max_horizons = [10, 10, 10]
            # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L46
            distance_thresholds = [[0.01, 0.02],   # We want to have precise subgoals
                                   [0.01, 0.02],
                                   [0.1, 10.0]]    # But for the goal I only care about the position (not the speed)

        action_noise_coeffs = np.array([0.1])  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L42
        subgoal_noise_coeffs = np.array([0.1, 0.1])  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L43
    elif env_name == "Pendulum":
        current_env = gym.make("Pendulum-v0")
        current_goal_state = np.array([0.0, 1.0, 0.0])

        # Action space: Low [-2.]	        High [2.]
        # State space:  Low [-1. -1. -8.]	High [1. 1. 8.]
        num_levels = 2
        max_horizons = [10, 25]
        distance_thresholds = [[0.10, 0.10, 1.0],  # Pendulum state = (x, y, angular velocity)
                               [0.05, 0.05, 0.4]]
        action_noise_coeffs = np.array([0.1])
        subgoal_noise_coeffs = np.array([0.02, 0.02, 0.5])
    elif env_name == "LunarLanderContinuous-v2":
        # Action space: Low [-1. -1.]	High [1. 1.]
        # State space:  Low [-inf] x 8         High [inf] x 8
        # State: x, y, vel.x, vel.y, lander.angle, angular_velocity, bool(left left on ground), bool(right leg on ground)
        current_env = gym.make("LunarLanderContinuous-v2")
        current_goal_state = np.array([0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0])

        num_levels = 2
        max_horizons = [10, 40]
        distance_thresholds = [[0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 2, 2],
                               [0.1, 0.02, 0.01, 0.01, 0.05, 0.02, 2, 2]]
        action_noise_coeffs = np.array([0.05, 0.05])
        subgoal_noise_coeffs = np.array([0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    else:
        raise Exception("Unsupported environment.")

    current_state_size = current_env.observation_space.low.shape[0]
    current_action_size = current_env.action_space.low.shape[0]

    ########################################
    #     Regularly changed parameters     #
    ########################################
    args = get_args()
    version = 3

    # current_directory = f"runs/{env_name}_{num_levels}_levels_h_{'_'.join(map(str, max_horizons))}_v{version}"
    current_directory = f"logs/{env_name}_{num_levels}_levels_h_{'_'.join(map(str, max_horizons))}_v{version}"
    random_id = None
    if args.run_on_cluster:
        random_id = str(random.randrange(1, 100000))
        dir_identifier = datetime.now().strftime('%b%d_%H-%M-%S') + '-' + random_id
        current_directory = os.environ['VSC_SCRATCH'] + '/' + current_directory + '/' + dir_identifier

    print(f"Current directory: {current_directory}")
    currently_training = True
    render_frequency = NEVER  # NEVER if args.no_render else FIRST_RUN

    if env_name == "LunarLanderContinuous-v2":
        num_training_episodes = 10000
    else:
        num_training_episodes = 2000

    evaluation_frequency = 30

    #############################
    #     Shared parameters     #
    #############################

    # batch_size=1024  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L43
    batch_size = 128  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L56

    discount = 0.98  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/master/critic.py#L8

    # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L54
    # Note: this parameters is actually more complicated than this, because the buffer size depends on the level
    # but currently, we're simplying it to a simple constant. TODO: see if this needs fixing
    # replay_buffer_size=10**7,  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L25
    replay_buffer_size = 500_000  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/117d4002e754a53019b5cf7f103946d382488217/utils.py#L4
    subgoal_testing_frequency = 0.3  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L125
    num_update_steps_when_training = 40  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/agent.py#L40

    current_hac_params = HacParams(
        action_low=current_env.action_space.low,
        action_high=current_env.action_space.high,
        state_low=current_env.observation_space.low,
        state_high=current_env.observation_space.high,
        batch_size=batch_size,
        num_training_episodes=num_training_episodes,
        num_levels=num_levels,
        max_horizons=max_horizons,
        discount=discount,
        replay_buffer_size=replay_buffer_size,
        subgoal_testing_frequency=subgoal_testing_frequency,
        distance_thresholds=distance_thresholds,
        action_noise_coeffs=action_noise_coeffs,
        subgoal_noise_coeffs=subgoal_noise_coeffs,
        num_update_steps_when_training=num_update_steps_when_training,
        evaluation_frequency=evaluation_frequency,
        save_frequency=evaluation_frequency,
        env_name=current_env.spec.id,
        random_id=str(random_id),
        run_on_cluster=args.run_on_cluster
    )

    print("Action space: Low %s\tHigh %s" % (current_env.action_space.low, current_env.action_space.high))
    print("State space: Low %s\tHigh %s" % (current_env.observation_space.low, current_env.observation_space.high))

    if currently_training:
        train(current_hac_params, current_env, current_goal_state, render_frequency, directory=current_directory)
    else:
        current_hac_params = load_hac(current_directory)
        evaluate_hac(current_hac_params, current_env, current_goal_state, render_frequency=ALWAYS, num_evals=100)

