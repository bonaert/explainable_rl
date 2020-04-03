from datetime import datetime

import numpy as np
import torch
import os
from math import ceil
import pickle as pkl
import gym

from tensorboardX import SummaryWriter

import hiro.utils as utils
import hiro.hiro as hiro

from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.common import get_reward_function

from hiro.eval_hiro import evaluate_policy


def run_hiro(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, args.log_file)):
        can_load = False
        os.makedirs(os.path.join(args.log_dir, args.log_file))
    else:
        can_load = True
        print("Existing directory found; may be able to load weights.")
    output_dir = os.path.join(args.log_dir, args.log_file)
    print("Logging in {}".format(output_dir))

    if args.env_name in ["MountainCarContinuous-v0", "LunarLanderContinuous-v2"]:
        env = EnvWithGoal(gym.make(args.env_name), args.env_name)
        # env.env.reward_type = args.reward_type
        if args.env_name == "MountainCarContinuous-v0":
            env.distance_threshold = 0.1
        elif args.env_name == "LunarLanderContinuous-v2":
            env.distance_threshold = 0.1
        min_obs, max_obs = env.base_env.observation_space.low, env.base_env.observation_space.high
        man_scale = max_obs - min_obs
        controller_goal_dim = man_scale.shape[0]
        no_xy = False  # Can't just take out first dimensions; movement here is different than for ants.
    elif "-v" in args.env_name:
        env = gym.make(args.env_name)
        env.env.reward_type = args.reward_type
        env.distance_threshold = env.env.distance_threshold
        max_action = np.array([1.54302745e+00, 1.21865324e+00, 9.98163424e-01, 1.97805133e-04,
                               7.15193042e-05, 2.56647627e-02, 2.30302501e-02, 2.13756120e-02,
                               1.19019512e-02, 6.31742249e-03])
        min_action = np.array(
            [7.95019864e-01, - 5.56192570e-02, 3.32176206e-01, 0.00000000e+00, 0.00000000e+00, - 2.58566763e-02,
             - 2.46581777e-02, - 1.77669761e-02, - 1.13476014e-02, - 5.08970149e-04])
        man_scale = max_action - min_action
        controller_goal_dim = man_scale.shape[0]
        no_xy = False  # Can't just take out first dimensions; movement here is different than for ants.
    else:
        # We'll be running on one of the various Ant envs
        env = EnvWithGoal(create_maze_env(args.env_name), args.env_name)

        # TODO: Where to these magic numbers come from?
        low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                        -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
        high = -low
        man_scale = (high - low) / 2
        controller_goal_dim = man_scale.shape[0]
        # scale = np.array([10, 10, 0.5, 1, 1, 1] + [60]*3 + [40]*3
        #                  + [60]*3 + [40]*3
        #                  + [60]*3 + [40]*3
        #                  + [60]*3 + [40]*3)
        no_xy = True

    obs = env.reset()

    goal = obs['desired_goal']
    state = obs['observation']

    # # Write Hyperparameters to file
    # print("---------------------------------------")
    # print("Current Arguments:")
    # with open(os.path.join(args.log_dir, args.log_file, "hps.txt"), 'w') as f:
    #     for arg in vars(args):
    #         print("{}: {}".format(arg, getattr(args, arg)))
    #         f.write("{}: {}\n".format(arg, getattr(args, arg)))
    # print("---------------------------------------\n")

    writer = SummaryWriter(logdir=os.path.join(args.log_dir, args.log_file))
    # torch.cuda.set_device(0)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    file_name = 'hiro_{}_{}'.format(args.env_name, current_time)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = state.shape[0]
    goal_dim = goal.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])

    # The goal dim is smaller than the state dim. This is very strange and doesn't seem to be compatible with
    # the paper and the formula g' = s + g - s' (since the states have different dimensions than the goals)
    # This works because the goal is a subpart of the state, so the update rule they actually use is:
    #    g' = s[:goal_dim] + g - s'[:goal_dim]
    # Initialize policy, replay buffers
    controller_policy = hiro.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        ctrl_rew_type=args.ctrl_rew_type,
        no_xy=no_xy,
    )

    manager_policy = hiro.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        should_reach_subgoal=args.should_reach_subgoal,
        subgoal_dist_cost_cf=args.man_subgoal_dist_cf
    )
    calculate_controller_reward = get_reward_function(controller_goal_dim)

    if args.noise_type == "ou":
        man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size)

    if can_load and args.load:
        try:
            manager_policy.load(output_dir)
            controller_policy.load(output_dir)
            manager_buffer.load(os.path.join(output_dir, "mbuf.npz"))
            controller_buffer.load(os.path.join(output_dir, "cbuf.npz"))
            with open(os.path.join(output_dir, "iter.pkl"), "rb") as f:
                iter = pkl.load(f) + 1
            print("Loaded successfully")
            just_loaded = True
        except Exception as e:
            iter = 0
            just_loaded = False
            print(e, "Not loading")
    else:
        iter = 0
        just_loaded = False

    # Logging Parameters
    total_timesteps = iter
    timesteps_since_eval = 0
    timesteps_since_manager = 0
    episode_timesteps = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    done = True
    evaluations = []

    while total_timesteps < args.max_timesteps:
        # Periodically save everything (controller, manager, buffers and total time steps)
        if args.save_every > 0 and (total_timesteps + 1) % args.save_every == 0:
            print("Saving")
            controller_policy.save(output_dir)
            manager_policy.save(output_dir)
            manager_buffer.save(os.path.join(output_dir, "mbuf.npz"))
            controller_buffer.save(os.path.join(output_dir, "cbuf.npz"))
            with open(os.path.join(output_dir, "iter.pkl"), "wb") as f:
                pkl.dump(total_timesteps, f)

        # If we finished the episode, we might have to (1) train the controller (2) evaluate the current policy
        # and (3) process final state/obs, store manager transition, if it was not just created
        # We train the controller at the end of every episode and the manager every X timesteps (not episodes!)
        if done:
            if total_timesteps != 0 and not just_loaded:
                # print('Training Controller...')
                ctrl_act_loss, ctrl_crit_loss = controller_policy.train(controller_buffer, episode_timesteps,
                                                                        args.ctrl_batch_size, args.ctrl_discount,
                                                                        args.ctrl_tau)

                print(ctrl_act_loss, ctrl_crit_loss, total_timesteps)
                writer.add_scalar('data/controller_actor_loss', ctrl_act_loss, total_timesteps)
                writer.add_scalar('data/controller_critic_loss', ctrl_crit_loss, total_timesteps)

                writer.add_scalar('data/controller_ep_rew', episode_reward, total_timesteps)
                writer.add_scalar('data/manager_ep_rew', manager_transition[4], total_timesteps)

                # Train Manager perdiocally
                if timesteps_since_manager >= args.train_manager_freq:
                    # print('Training Manager...')
                    timesteps_since_manager = 0
                    man_act_loss, man_crit_loss = manager_policy.train(
                        controller_policy,
                        manager_buffer,
                        ceil(episode_timesteps / args.train_manager_freq),
                        args.man_batch_size, args.discount,
                        args.man_tau
                    )

                    writer.add_scalar('data/manager_actor_loss', man_act_loss, total_timesteps)
                    writer.add_scalar('data/manager_critic_loss', man_crit_loss, total_timesteps)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish = evaluate_policy(
                        env, writer, manager_policy, controller_policy, calculate_controller_reward,
                        args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations),
                        render=args.render_in_eval
                    )

                    writer.add_scalar('eval/avg_ep_rew', avg_ep_rew, total_timesteps)
                    writer.add_scalar('eval/avg_controller_rew', avg_controller_rew, total_timesteps)
                    writer.add_scalar('eval/avg_steps_to_finish', avg_steps, total_timesteps)
                    writer.add_scalar('eval/perc_env_goal_achieved', avg_env_finish, total_timesteps)

                    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])

                    if args.save_models:
                        controller_policy.save(file_name + '_controller', directory="./pytorch_models")
                        manager_policy.save(file_name + '_manager', directory="./pytorch_models")

                    np.save("./results/%s" % file_name, evaluations)

                # Process final state/obs, store manager transition, if it was not just created
                if len(manager_transition[-2]) != 1:  # If there's more than 1 state in the transition
                    # Manager transitions are a list of the form
                    # [initial state, final state, goal, subgoal, manager reward, done, states, actions]
                    manager_transition[1] = state  # Store the final state
                    manager_transition[5] = float(True)  # Set done to true

                    # Every manager transition should have same length of sequences
                    # TODO: this suggests one difficulty of not changing subgoals with a specific period:
                    # TODO: the input dimensions will vary! figure out how to deal with this
                    if len(manager_transition[-2]) <= args.manager_propose_freq:
                        while len(manager_transition[-2]) <= args.manager_propose_freq:
                            manager_transition[-1].append(np.inf)
                            manager_transition[-2].append(state)

                    manager_buffer.add(manager_transition)

            # Reset environment
            obs = env.reset()
            goal = obs['desired_goal']
            state = obs['observation']

            done = False
            episode_reward = 0
            episode_timesteps = 0
            just_loaded = False
            episode_num += 1

            # Create new manager transition (sample new subgoal)
            subgoal = manager_policy.sample_subgoal(state, goal)

            timesteps_since_subgoal = 0

            # Create a high level transition
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

        # TODO: Scale action to environment
        action = controller_policy.select_action(state, subgoal)
        action = ctrl_noise.perturb_action(action, max_action)

        # Perform action, get (nextst, r, d)
        next_tup, manager_reward, env_done, _ = env.step(action)

        # Update cumulative reward (env. reward) for manager
        manager_transition[4] += manager_reward * args.man_rew_scale

        # Process
        next_goal = next_tup['desired_goal']
        next_state = next_tup['observation']

        # Append low level sequence for off policy correction
        manager_transition[-1].append(action)
        manager_transition[-2].append(next_state)

        # Calculate reward, transition subgoal
        # print(np.sum(np.abs(state - next_state)), subgoal)

        controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
        subgoal = controller_policy.subgoal_transition(state, subgoal, next_state)

        controller_goal = subgoal
        # Is the episode over?
        if env_done:
            done = True

        episode_reward += controller_reward

        # Store low level transition
        if args.inner_dones:
            ctrl_done = done or timesteps_since_subgoal % args.manager_propose_freq == 0
        else:
            ctrl_done = done
        controller_buffer.add((state, next_state, controller_goal, action, controller_reward, float(ctrl_done), [], []))

        # Update state parameters
        state = next_state
        goal = next_goal

        # Update counters
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        # Every X timesteps, store manager transition in buffer and pick a new subgoal
        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            # Finish, add transition
            manager_transition[1] = state
            manager_transition[5] = float(done)

            manager_buffer.add(manager_transition)

            subgoal = manager_policy.sample_subgoal(state, goal)
            subgoal = man_noise.perturb_action(subgoal, max_action=man_scale)

            # Reset number of timesteps since we sampled a subgoal
            timesteps_since_subgoal = 0

            # Create a high level transition
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

    # Final evaluation
    evaluations.append([evaluate_policy(env, writer, manager_policy, controller_policy,
                                        calculate_controller_reward, args.ctrl_rew_scale,
                                        args.manager_propose_freq, len(evaluations))])

    if args.save_models:
        controller_policy.save(file_name + '_controller', directory="./pytorch_models")
        manager_policy.save(file_name + '_manager', directory="./pytorch_models")

    np.save("./results/%s" % file_name, evaluations)
