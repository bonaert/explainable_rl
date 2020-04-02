import os

import gym
import torch
import numpy as np
from hiro import hiro

from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.common import get_reward_function


def get_env_and_policy(args):
    # Load environment
    if "-v" in args.env_name:
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
    # Fetch environment meta info
    obs = env.reset()
    goal = obs['desired_goal']
    state = obs['observation']
    state_dim = state.shape[0]
    goal_dim = goal.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
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
    # Reload weights from file
    output_dir = os.path.join(args.log_dir, args.log_file)
    manager_policy.load(output_dir)
    controller_policy.load(output_dir)
    calculate_controller_reward = get_reward_function(controller_goal_dim)
    return env, controller_policy, manager_policy, calculate_controller_reward


def eval_hiro(args):
    env, controller_policy, manager_policy, calculate_controller_reward = get_env_and_policy(args)

    evaluate_policy(env=env,
                    writer=None,
                    manager_policy=manager_policy,
                    controller_policy=controller_policy,
                    calculate_controller_reward=calculate_controller_reward,
                    ctrl_rew_scale=args.ctrl_rew_scale,
                    manager_propose_frequency=args.manager_propose_freq,
                    eval_episodes=1000,
                    log_every_episode=True,
                    render=True)


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, writer, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_idx=0, eval_episodes=5,
                    log_every_episode=False, render=False):
    print("Starting evaluation number {}...".format(eval_idx))
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        for eval_ep in range(eval_episodes):
            obs = env.reset()

            goal = obs['desired_goal']
            state = obs['observation']

            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    print("Step %d: The manager sampled a goal!" % step_count)
                    subgoal = manager_policy.sample_subgoal(state, goal)

                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(state, subgoal)
                new_obs, reward, done, _ = env.step(action)
                if render:
                    env.base_env.render(mode='human')
                # See if the environment goal was achieved
                goal_achieved = reward >= -env.distance_threshold
                if goal_achieved:
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True

                if log_every_episode and done:
                    print("---------------------------------------")
                    print("Goal achieved: %s" % goal_achieved)
                    print("Number of steps: %s" % step_count)
                    print("Evaluation over {} episodes: {:.3f} \nAvg Ctrl Reward: {:.3f}".format(
                        eval_ep + 1, avg_reward / (eval_ep + 1), avg_controller_rew / global_steps))
                    print(
                        "Goals achieved: {:.2f}% ({} / {})".format(100 * goals_achieved / (eval_ep + 1), goals_achieved,
                                                                   eval_ep + 1))
                    print('Average Steps to finish: {:.3f}'.format(global_steps / (eval_ep + 1)))
                    print("---------------------------------------")

                goal = new_obs['desired_goal']
                new_state = new_obs['observation']

                # Update subgoal in the controller (g' = s + g - s'). DOESN'T update the subgoal in the manager
                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                controller_reward = calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)
                print(controller_reward)
                avg_controller_rew += controller_reward

                state = new_state

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes: {} \nAvg Ctrl Reward: {}".format(eval_episodes, avg_reward,
                                                                             avg_controller_rew))
        print("Goals achieved: {}%".format(100 * goals_achieved / eval_episodes))
        print('Average Steps to finish: {}'.format(avg_step_count))
        print("---------------------------------------")

        env.evaluate = False
        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, writer, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_idx=0, eval_episodes=5,
                    log_every_episode=False, render=False):
    print("Starting evaluation number {}...".format(eval_idx))
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        for eval_ep in range(eval_episodes):
            obs = env.reset()

            goal = obs['desired_goal']
            state = obs['observation']

            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    print("Step %d: The manager sampled a goal!" % step_count)
                    subgoal = manager_policy.sample_subgoal(state, goal)

                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(state, subgoal)
                new_obs, reward, done, _ = env.step(action)
                if render:
                    env.base_env.render(mode='human')
                # See if the environment goal was achieved
                goal_achieved = reward >= -env.distance_threshold
                if goal_achieved:
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True

                if log_every_episode and done:
                    print("---------------------------------------")
                    print("Goal achieved: %s" % goal_achieved)
                    print("Number of steps: %s" % step_count)
                    print("Evaluation over {} episodes: {:.3f} \nAvg Ctrl Reward: {:.3f}".format(
                        eval_ep + 1, avg_reward / (eval_ep + 1), avg_controller_rew / global_steps))
                    print(
                        "Goals achieved: {:.2f}% ({} / {})".format(100 * goals_achieved / (eval_ep + 1), goals_achieved,
                                                                   eval_ep + 1))
                    print('Average Steps to finish: {:.3f}'.format(global_steps / (eval_ep + 1)))
                    print("---------------------------------------")

                goal = new_obs['desired_goal']
                new_state = new_obs['observation']

                # Update subgoal in the controller (g' = s + g - s'). DOESN'T update the subgoal in the manager
                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                controller_reward = calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)
                print(controller_reward)
                avg_controller_rew += controller_reward

                state = new_state

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes: {} \nAvg Ctrl Reward: {}".format(eval_episodes, avg_reward,
                                                                             avg_controller_rew))
        print("Goals achieved: {}%".format(100 * goals_achieved / eval_episodes))
        print('Average Steps to finish: {}'.format(avg_step_count))
        print("---------------------------------------")

        env.evaluate = False
        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish
