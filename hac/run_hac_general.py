import gym
import numpy as np
from hac_general import HacParams, evaluate_hac, train, load_hac
from common import ALWAYS, FIRST_RUN, NEVER

if __name__ == '__main__':
    # noinspection PyUnreachableCode
    ##################################
    #     Environment parameters     #
    ##################################
    # env_name = "AntMaze"
    # env_name = "MountainCar"
    env_name = "Pendulum"
    if env_name == "AntMaze":
        # state_distance_thresholds = [0.1, 0.1]  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L106
        # max_horizons = 10  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L27
        # action_noise_coeffs = np.array([0.1] * current_action_size),  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L131
        # subgoal_noise_coeffs = np.array([0.03] * current_state_size),  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L132
        raise Exception("TODO")
    elif env_name == "MountainCar":
        current_env = gym.make("MountainCarContinuous-v0")

        # noinspection PyUnreachableCode
        if False:
            num_levels = 2
            max_horizons = [20]  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L50
            # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L46
            state_distance_thresholds = [[0.1, 0.1]]
        else:
            num_levels = 3
            max_horizons = [20, 15]
            # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L46
            state_distance_thresholds = [[0.1, 0.1],  # We want to have precise subgoals
                                         [0.1, 0.1]]    # But for the goal I only care about the position (not the speed)

        action_noise_coeffs = np.array([0.1])  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L42
        state_noise_coeffs = np.array([0.2, 0.2])  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L43
        reward_noise_coeff = 1
        reward_low = -100.0  # TODO
        reward_high = 100.0   # TODO
        current_env_threshold = current_env.spec.reward_threshold
    elif env_name == "Pendulum":
        current_env = gym.make("Pendulum-v0")

        # Action space: Low [-2.]	        High [2.]
        # State space:  Low [-1. -1. -8.]	High [1. 1. 8.]
        num_levels = 2
        max_horizons = [5]
        state_distance_thresholds = [[0.1, 0.1, 1]]  # Pendulum state + reward = (x, y, angular velocity)
        action_noise_coeffs = np.array([0.1])
        state_noise_coeffs = np.array([0.02, 0.02, 0.1])
        reward_noise_coeff = 2
        reward_low = -15 * max_horizons[0]
        reward_high = 5 * max_horizons[0]
        current_env_threshold = -150.0
    else:
        raise Exception("Unsupported environment.")

    current_state_size = current_env.observation_space.low.shape[0]
    current_action_size = current_env.action_space.low.shape[0]

    ########################################
    #     Regularly changed parameters     #
    ########################################
    version = 4
    current_directory = f"{env_name}_{num_levels}_hac_general_levels_h_{'_'.join(map(str, max_horizons))}_v{version}"
    currently_training = True
    my_render_frequency = FIRST_RUN
    num_training_episodes = 5000
    evaluation_frequency = 50

    print("Action space: Low %s\tHigh %s" % (current_env.action_space.low, current_env.action_space.high))
    print("State space: Low %s\tHigh %s" % (current_env.observation_space.low, current_env.observation_space.high))

    if currently_training:
        #############################
        #     Shared parameters     #
        #############################

        # batch_size=1024  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L43
        current_batch_size = 128  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L56

        # discount=0.98  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/master/critic.py#L8
        current_discount = 0.95

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
            reward_low=reward_low,
            reward_high=reward_high,
            batch_size=current_batch_size,
            num_training_episodes=num_training_episodes,
            num_levels=num_levels,
            max_horizons=max_horizons,
            discount=current_discount,
            replay_buffer_size=replay_buffer_size,
            subgoal_testing_frequency=subgoal_testing_frequency,
            state_distance_thresholds=state_distance_thresholds,
            action_noise_coeffs=action_noise_coeffs,
            state_noise_coeffs=state_noise_coeffs,
            reward_noise_coeff=reward_noise_coeff,
            num_update_steps_when_training=num_update_steps_when_training,
            evaluation_frequency=evaluation_frequency,
            save_frequency=evaluation_frequency,
            env_threshold=current_env_threshold
        )

        train(current_hac_params, current_env, my_render_frequency, directory=current_directory)
    else:
        current_hac_params = load_hac(current_directory)
        evaluate_hac(current_hac_params, current_env, render_frequency=ALWAYS, num_evals=100)
