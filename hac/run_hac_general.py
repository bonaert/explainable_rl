import gym
import numpy as np
from hac_general import HacParams, evaluate_hac, train, load_hac
from common import get_args, ActionRepeatEnvWrapper
from training.sac import get_policy_and_scaler

if __name__ == '__main__':
    # noinspection PyUnreachableCode
    ##################################
    #     Environment parameters     #
    ##################################

    args = get_args()
    if args.env_name == "NotProvided":
        # env_name = "AntMaze"
        # env_name = "MountainCar"
        # env_name = "Pendulum"
        # env_name = "BipedalWalker-v3"
        env_name = "LunarLanderContinuous-v2"
    else:
        env_name = args.env_name

    overriden_state_space_low = None
    overriden_state_space_high = None
    teacher = None

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
        reward_low = [-100.0] * num_levels  # TODO
        reward_high = [100.0] * num_levels  # TODO
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
        reward_low = [-15 * max_horizons[0]] * num_levels
        reward_high = [5 * max_horizons[0]] * num_levels
        current_env_threshold = -150.0
        penalty_subgoal_reachability = -2000
    elif env_name == "BipedalWalker-v3":
        # Action space: Low [-1. -1. -1. -1.]	High [1. 1. 1. 1.]
        # State space:  Low [-inf] x 24         High [inf] x 24
        #               14 values (speed, angle joints) + 10 lidar values
        reward_scale = 1.0
        current_env = gym.make('BipedalWalker-v3')
        # current_env = ActionRepeatEnvWrapper(current_env, action_repeat=3, reward_scale=reward_scale)
        num_levels = 2
        max_horizons = [10]

        # State spaces we can pick goals from: [-10, 10] for values and [-100, 100] for lidar values
        overriden_state_space_low = np.array([-5] + [-2] * 13 + [-2] * 10, dtype=np.float32)
        overriden_state_space_high = np.array([5] + [2] * 13 + [2] * 10, dtype=np.float32)
        # Very small distance threshold for values; don't try to predict lidar values (probably impossible)
        state_distance_thresholds = [[0.5] * 14 + [np.inf] * 10]

        # Not used with SAC
        action_noise_coeffs = np.array([0.5] * 4)
        state_noise_coeffs = np.array([0.05] * 24)
        reward_noise_coeff = 0.3

        # reward_low = [-200, -200]
        # reward_high = [350, 350]
        reward_low = [None, -5 * max_horizons[0]]
        reward_high = [None, 5 * max_horizons[0]]

        current_env_threshold = 300.0 * reward_scale
    elif env_name == "LunarLanderContinuous-v2":
        # Action space: Low [-1. -1.]	High [1. 1.]
        # State space:  Low [-inf] x 8         High [inf] x 8
        # State: x, y, vel.x, vel.y, lander.angle, angular_velocity, bool(left left on ground), bool(right leg on ground)
        reward_scale = 1.0 / 100
        current_env = gym.make('LunarLanderContinuous-v2')
        current_env = ActionRepeatEnvWrapper(current_env, action_repeat=1, reward_scale=reward_scale)
        num_levels = 2
        max_horizons = [10]

        current_goal_state = np.array([0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0])

        overriden_state_space_low = np.array([-2, -5, -3, -3, -10, -10, 0, 0], dtype=np.float32)
        overriden_state_space_high = np.array([2,  5,  3,  3,  20,  10, 1, 1], dtype=np.float32)

        state_distance_thresholds = [[0.04, 0.05, 0.5, 0.2, 1.0, 1, 0.5, 0.5]]

        # Not used with SAC
        action_noise_coeffs = np.array([0.5] * 2)
        state_noise_coeffs = np.array([0.05] * 8)
        reward_noise_coeff = 0.3

        # reward_low = [-200, -200]
        # reward_high = [350, 350]
        reward_low = [None, -1000 * reward_scale]
        reward_high = [None, 200 * reward_scale]

        current_env_threshold = 200.0 * reward_scale
        penalty_subgoal_reachability = -1000.0 * reward_scale

        # Teacher stuff
        teacher, scaler = get_policy_and_scaler(current_env, has_scaler=True)
        probability_to_use_teacher = 0.5
    else:
        raise Exception("Unsupported environment.")

    current_state_size = current_env.observation_space.low.shape[0]
    current_action_size = current_env.action_space.low.shape[0]

    ########################################
    #     Regularly changed parameters     #
    ########################################

    # It learns to pick reachable subgoals
    # But it doesn't pick subgoals that maximize the reward well
    # Maybe it picks subgoals that are reachable but bad, and it stays stuck there
    # E.g. it has bad exploration :(

    use_sac = True
    use_priority_replay = False
    version = 12
    current_directory = f"runs/{env_name}_{'sac' if use_sac else 'ddpg'}_{num_levels}_hac_general_levels_h_{'_'.join(map(str, max_horizons))}_v{version}"
    print(f"Current directory: {current_directory}")
    currently_training = not args.test
    num_training_episodes = args.num_training_episodes
    evaluation_frequency = args.eval_frequency
    my_render_rounds = args.render_rounds
    current_num_test_episodes = args.current_num_test_episodes
    all_levels_maximize_reward = args.all_levels_maximize_reward
    reward_present_in_input = args.reward_present_in_input

    current_batch_size = args.batch_size
    current_discount = args.discount
    replay_buffer_size = args.replay_buffer_size
    subgoal_testing_frequency = args.subgoal_testing_frequency
    num_update_steps_when_training = args.num_update_steps_when_training
    learning_rates = [3e-4, 3e-4]

    print("Action space: Low %s\tHigh %s" % (current_env.action_space.low, current_env.action_space.high))
    print("State space: Low %s\tHigh %s" % (current_env.observation_space.low, current_env.observation_space.high))

    if currently_training:
        #############################
        #     Shared parameters     #
        #############################

        # batch_size=1024  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L43
        # current_batch_size = 128  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L56
        #
        # # discount=0.98  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/master/critic.py#L8
        # current_discount = 0.95
        #
        # # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/train.py#L54
        # # Note: this parameters is actually more complicated than this, because the buffer size depends on the level
        # # but currently, we're simplying it to a simple constant. TODO: see if this needs fixing
        # # replay_buffer_size=10**7,  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L25
        # replay_buffer_size = 2_000_000  # https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/117d4002e754a53019b5cf7f103946d382488217/utils.py#L4
        # subgoal_testing_frequency = 0.2  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/design_agent_and_env.py#L125
        # # num_update_steps_when_training = 40  # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/agent.py#L40
        # num_update_steps_when_training = 40

        current_hac_params = HacParams(
            action_low=current_env.action_space.low,
            action_high=current_env.action_space.high,
            state_low=overriden_state_space_low if overriden_state_space_low is not None else current_env.observation_space.low,
            state_high=overriden_state_space_high if overriden_state_space_high is not None else current_env.observation_space.high,
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
            env_threshold=current_env_threshold,
            env_name=current_env.spec.id,
            use_priority_replay=use_priority_replay,
            penalty_subgoal_reachability=penalty_subgoal_reachability,
            use_sac=use_sac,
            all_levels_maximize_reward=all_levels_maximize_reward,
            reward_present_in_input=reward_present_in_input,
            num_test_episodes=current_num_test_episodes,
            goal_state=current_goal_state,
            learning_rates=learning_rates,

            # Teacher suff
            teacher=teacher,
            state_scaler=scaler,
            probability_to_use_teacher=probability_to_use_teacher
        )

        train(current_hac_params, current_env, my_render_rounds, directory=current_directory)
    else:
        current_hac_params = load_hac(current_directory)
        num_successes, success_rate, rewards, steps_per_episode = evaluate_hac(current_hac_params, current_env, render_rounds=100000 if not args.test else 0, num_evals=100)
        print("\nSuccess rate (%d/%d): %.3f" % (num_successes, len(rewards), success_rate))
        # noinspection PyStringFormat
        print("Reward: %.3f +- %.3f" % (np.mean(rewards), np.std(rewards)))
        # noinspection PyStringFormat
        print("Number of steps: %.3f +- %.3f" % (np.mean(steps_per_episode), np.std(steps_per_episode)))
