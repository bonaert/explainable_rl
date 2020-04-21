import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional, Union

import numpy as np
from pathlib import Path

import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import get_range_and_center, json_default, FIRST_RUN, ALWAYS
from ddpg import DDPG
from sac import Sac

HUGE_PENALTY = -500


# I think we need special logic for the top level agent


# I need to be clear about what each agent tries to do.
# What does the actor predict as actions and what does it take as input?
#  - Top level: (state) -> subgoal = (desired state, desired reward)
#  - Middle levels: (state, goal) -> subgoals = (desired state, desired reward)
#  - Low level: (state, goal) -> raw actions
# What is the critic evaluating?
#  - Top level: (state, action) -> Q-value
#  - Middle levels: (state, action, goal) -> Q-value
#  - Low level: (state, action, goal) -> Q-value

# Here are the transitions that we use
#
# Top level (with no upper goal to use):
# - Hindsight action transitions, where we land on s' and get env reward r (s, a=s', r, s', discount)
# - No hindsight goal transitions, since there is no goal from above
# - Subgoal testing transitions. When the lower level policy fails to reach the subgoal/action, we create a transition (s, a, -10000, s', 0)
#
# Middle levels (with goal as input)
# - Hindsight action transitions, same as top level, but with reward/discount either (0, 0) or (-1, discount) depending on if the reached
#   the goal from above or not
# - Hindsight goal transitions. Pick an end state s among H hindsight action transitions and then replace the goal by s in all H transitions
# - Subgoal testing transitions. Use (s, a, r, s', discount=0) where r = 0 if we reached the subgoal otherwise -H
#
# Bottom level:
# - The hindsight action is just the action, we don't replace it.
# - Hindsight goal transitions work at in the middle levels
# - No subgoal testing transitions, since the action are environment action and not subgoals.


#
# The question is: do the layer below get Q-values in the range [-H, 0] as in normal HAC or do we keep track of the real rewards
# at every layer? I don't think it's strictly necessary to use the real rewards, because we're trying to predict the real reward.
# Actors will try to pick actions that maximize the Q-values, so these Q-values have to reward accuracy for the non-top level layers.
# As long as failing to hit it leads to bad reward and hitting the subgoal lead to rewards, we should be good. So I think for
# subgoal-reaching layer we can keep [-H, 0]. But would using the real reward better? This would mean that reaching a subgoal
# with a higher reward would be incentivized than reaching a subgoal with a lower reward. However, not reaching the subgoal would be
# equally punished for both.
# On the other hand, the top level layer needs to

# Goals and subgoals are always pairs (desired state, desired reward)
# On the other hand, the action isn't always a subgoal, sadly...
# It would be nice if the goals and subgoals are always pairs
# On the other hand, the action can either be:
# 1) a low-level action
# 2) a subgoal (desired state, desired reward)
# Again, I think encoding this a single thing would be easier, and then we'd check the level we're at to see how to
# handle the action details when it's needed
# And should a subgoal be a tuple or numpy array? I think it's probably easier if it's a numpy array


# There's a problem when I try the 3 level agent: it starts picking subgoals that are immediately reached.
# I probably need to add transitions that punish this kind of behavior.

def mk_transition(state: np.ndarray, action: np.ndarray, env_reward: float, total_env_reward: float,
                  next_state: np.ndarray, transition_reward: float, goal: np.ndarray, discount: float):
    """ This function is here just for type-checking and making it easier to ensure the order is correct through the IDE help """
    # Env reward: the reward given by the environment after doing the action
    # Total env reward: the sum of the rewards given by the environment after all the actions the agent has done
    # Transition reward: the reward we chose for this transition, which may be different than the reward given by the environment
    #        0       1         2             3              4              5            6       7
    return state, action, env_reward, total_env_reward, next_state, transition_reward, goal, discount


@dataclass
class HacParams:

    action_low: np.ndarray
    action_high: np.ndarray

    state_low: np.ndarray
    state_high: np.ndarray

    reward_low: float
    reward_high: float

    batch_size: int
    num_training_episodes: int
    num_levels: int
    max_horizons: List[int]
    discount: float
    replay_buffer_size: int
    subgoal_testing_frequency: float
    # If the distance between the state and the goal is below this threshold, we consider that we reached the goal
    state_distance_thresholds: List[List[float]]
    # When adding Normal noise to the actions, we multiply the (high - low) / 2 by these
    # coefficients to ge the desired standard deviation
    action_noise_coeffs: np.ndarray
    state_noise_coeffs: np.ndarray
    reward_noise_coeff: float

    num_update_steps_when_training: int

    evaluation_frequency: int

    save_frequency: int

    env_threshold: float

    env_name: str

    # These fields have a default value but the user should be able
    # to override them.
    use_sac: bool = False  # By default we use DDPG, but we can switch to SAC
    use_tensorboard: bool = True
    step_number: int = 0

    # Fields with default value that will be filled with a true value in the __post_init__ method
    # Important: The user shouldn't fill these themselves! The values will be overwritten.
    state_size: int = -1
    action_size: int = -1

    # These are going to be overwritten anyway, so I can just initialize them to None
    action_range: np.ndarray = field(default_factory=lambda: None)
    action_center: np.ndarray = field(default_factory=lambda: None)

    subgoal_space_low: np.ndarray = field(default_factory=lambda: None)
    subgoal_space_high: np.ndarray = field(default_factory=lambda: None)
    subgoal_range: np.ndarray = field(default_factory=lambda: None)
    subgoal_center: np.ndarray = field(default_factory=lambda: None)
    subgoal_noise_coeffs: np.ndarray = field(default_factory=lambda: None)

    her_storage: List[List[list]] = field(default_factory=list)
    policies: List[Union[Sac, DDPG]] = field(default_factory=list)

    writer: SummaryWriter = field(default_factory=lambda: None)  # Tensorboard writer

    def __post_init__(self):
        # This method is executed at the end of the constructor. Here, I can setup the list I need
        # I do some validation then setup some variables with their real value
        # This is useful for the user, which doesn't have to it themselves and saves work
        # It also ensures it's done correctly
        assert 0 <= self.subgoal_testing_frequency <= 1, "Subgoal testing frequency must be between 0 and 1"
        assert 0 <= self.discount <= 1, "Discount must be between 0 and 1"
        assert 1 <= self.num_levels, "The number of levels must be at least 1"
        assert (1 <= np.array(self.max_horizons)).all(), "All horizons must at least be 1 step long"
        assert len(self.max_horizons) == self.num_levels - 1, \
            "There are %d horizons for the non-top level(s) but there are %d non-top level(s)" % (len(self.max_horizons), self.num_levels - 1)
        assert self.reward_low <= self.reward_high, "Reward low is larger than reward high"
        assert len(self.state_distance_thresholds) == self.num_levels - 1, \
            "Number of distances thresholds (%d) is different from the number of non-top levels (%d)" % (
                len(self.state_distance_thresholds), self.num_levels - 1)
        assert not np.isinf(self.action_low).any(), "Error: the action space cannot have +-infinite lower bounds"
        assert not np.isinf(self.action_high).any(), "Error: the action space cannot have +-infinite upper bounds"
        assert not np.isinf(self.state_low).any(), "Error: the state space cannot have +-infinite lower bounds"
        assert not np.isinf(self.state_high).any(), "Error: the state space cannot have +-infinite upper bounds"

        self.subgoal_space_low = np.hstack([self.state_low, self.reward_low])
        self.subgoal_space_high = np.hstack([self.state_high, self.reward_high])

        self.action_range, self.action_center = get_range_and_center(self.action_low, self.action_high)
        self.subgoal_range, self.subgoal_center = get_range_and_center(self.subgoal_space_low, self.subgoal_space_high)

        self.state_size = len(self.state_low)
        self.action_size = len(self.action_low)
        self.subgoal_size = self.state_size + 1

        self.subgoal_noise_coeffs = np.hstack([self.state_noise_coeffs, self.reward_noise_coeff])

        for i in range(self.num_levels - 1):
            assert len(self.state_distance_thresholds[i]) == self.state_size, \
                "Number of distances thresholds at level %d is %d but should be %d (state dim)" % (
                    i, len(self.state_distance_thresholds[i]), self.state_size)

        assert len(self.subgoal_noise_coeffs) == self.subgoal_size, \
            "Subgoal noise has %d dims but the states have %d dims" % (len(self.subgoal_noise_coeffs), self.subgoal_size)
        assert len(self.action_noise_coeffs) == self.action_size, \
            "Action noise has %d dims but the actions have %d dims" % (len(self.action_noise_coeffs), self.action_size)

        if self.use_tensorboard:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.writer = SummaryWriter(f"logs/{self.env_name}/{current_time}")

        self.her_storage = [[] for _ in range(self.num_levels)]
        self.policies = []
        for level in range(self.num_levels):
            if self.use_sac:
                agent = Sac(
                    state_size=self.state_size,
                    goal_size=self.subgoal_size if level != self.num_levels - 1 else 0,  # No goal for the top level
                    action_low=self.subgoal_space_low if level > 0 else self.action_low,
                    action_high=self.subgoal_space_high if level > 0 else self.action_high,
                    q_bound=-self.max_horizons[level] if level < self.num_levels - 1 else None,  # [-H, 0] Q-values for non-top levels
                    buffer_size=self.replay_buffer_size,
                    batch_size=self.batch_size,
                    writer=self.get_tensorboard_writer() if self.use_tensorboard else None,
                    sac_id='Level %d' % level
                )
            else:
                agent = DDPG(
                    state_size=self.state_size,
                    goal_size=self.subgoal_size if level != self.num_levels - 1 else 0,  # No goal for the top level
                    action_range=self.subgoal_range if level > 0 else self.action_range,
                    action_center=self.subgoal_center if level > 0 else self.action_center,
                    q_bound=-self.max_horizons[level] if level < self.num_levels - 1 else None,  # [-H, 0] Q-values for non-top levels
                    buffer_size=self.replay_buffer_size,
                    batch_size=self.batch_size
                )
            self.policies.append(agent)

    def is_top_level(self, level: int) -> bool:
        return level == self.num_levels - 1

    def get_tensorboard_writer(self) -> SummaryWriter:
        """ Returns a Tensorboard writer, which allows logging many types of information (scalars, images, ...)"""
        return self.writer


def reached_subgoal(state: np.ndarray, env_reward: float, goal: np.ndarray, level: int, hac_params: HacParams) -> bool:
    desired_state, desired_reward = goal[:-1], goal[-1]
    distances = np.abs(state - desired_state)
    return env_reward >= desired_reward and (distances < hac_params.state_distance_thresholds[level]).all()


def reached_any_supergoal(current_state: np.ndarray, env_reward: float, subgoals_stack: List[np.ndarray], level: int, hac_params: HacParams):
    for subgoal in subgoals_stack:
        if reached_subgoal(current_state, env_reward, subgoal, level, hac_params):
            return True

    return False


def compute_transition_reward_and_discount(state: np.ndarray, env_reward: float, goal: np.ndarray, level: int, done: bool, hac_params: HacParams) -> Tuple[float, float]:
    if hac_params.is_top_level(level):
        reward, discount = env_reward, (1.0 - float(done)) * hac_params.discount
    else:
        if reached_subgoal(state, env_reward, goal, level, hac_params):
            reward, discount = 0.0, 0.0
        else:
            reward, discount = -1.0, hac_params.discount

    return reward, discount


def perform_HER(her_storage: List[list], level: int, hac_params: HacParams) -> List[tuple]:
    if len(her_storage) == 0:  # Can happen if we're directly at a subgoal
        return []

    transitions = her_storage[:]  # Make a copy to be sure we don't fuck things up
    completed_transitions = []

    # "First, one of the “next state” elements in one of the transitions will be selected
    #  as the new goal state replacing the TBD component in each transition"
    random_transition = random.choice(transitions)
    chosen_goal = np.hstack([random_transition[4], random_transition[3]])
    #                            next_state          total_env_reward

    for transition in transitions:
        # We need to update the transition reward (5), the goal (6) and discount (7)
        # goal_transition = (current_state, action, env_reward, total_env_reward, next_state, None, None, None, done)
        tr_next_state = transition[4]
        tr_total_env_reward = transition[3]
        # The done parameter is irrelevant, since we don't perform HER for the top level
        reward, discount = compute_transition_reward_and_discount(tr_next_state, tr_total_env_reward, chosen_goal, level, False, hac_params)
        transition[5] = reward
        transition[6] = chosen_goal
        transition[7] = discount

        completed_transitions.append(tuple(transition))

    return completed_transitions


def get_random_action(level: int, env: gym.Env, hac_params: HacParams) -> np.ndarray:
    if level == 0:
        return np.random.uniform(env.action_space.low, env.action_space.high)
    else:
        return np.random.uniform(hac_params.subgoal_space_low, hac_params.subgoal_space_high)


def add_noise(action: np.ndarray, level: int, env: gym.Env, hac_params: HacParams) -> np.ndarray:
    if level == 0:
        action_low, action_high = env.action_space.low, env.action_space.high
    else:
        action_low, action_high = hac_params.subgoal_space_low, hac_params.subgoal_space_high

    noise_coeff = hac_params.action_noise_coeffs if level == 0 else hac_params.subgoal_noise_coeffs
    action += np.random.normal(0, noise_coeff)  # I'm using the Pytorch's implementation, it's different in the original TF one
    action = np.clip(action, action_low, action_high)
    return action


def pick_action_and_testing(state: np.ndarray, goal: np.ndarray, level: int, is_testing_subgoal: bool,
                            env: gym.Env, hac_params: HacParams, training: bool) -> Tuple[np.ndarray, bool]:
    # If the layer above was testing, it requires that everything below it have deterministic (non-noisy)
    # behavior too. Therefore, this level must also be deterministic and be in "testing subgoal" mode
    # where we don't add any noise. Additionally, if we're not training but only evaluating the policy, we don't add noise.
    if is_testing_subgoal or not training:
        action = hac_params.policies[level].sample_action(state, goal, deterministic=True)
        return action, True

    # Exploration: Each level uses the following exploration strategy when a level is not involved in subgoal testing.
    # 20% of actions are sampled uniformly at random from the level’s action space
    # 80% of actions are the sum of actions sampled from the level’s policy and Gaussian noise
    if random.random() < 0.2:
        action = get_random_action(level, env, hac_params)
    else:
        if hac_params.use_sac:
            action = hac_params.policies[level].sample_action(state, goal, deterministic=False)
        else:
            action = hac_params.policies[level].sample_action(state, goal)
            action = add_noise(action, level, env, hac_params)

    # We start testing a certain fraction lambda of the time, e.g. with a probability lambda
    if random.random() < hac_params.subgoal_testing_frequency:
        is_testing_subgoal = True
    else:
        is_testing_subgoal = False

    return action, is_testing_subgoal


def run_HAC_level(level: int, start_state: np.ndarray, goal: np.ndarray,
                  env: gym.Env, hac_params: HacParams,
                  is_testing_subgoal: bool, subgoals_stack: List[np.ndarray],
                  training: bool, render: bool) -> Tuple[np.ndarray, float, bool, bool]:
    """ Returns (last state, collected reward, if it failed to reach the goal, done)"""
    current_state = start_state
    num_attempts = 0
    total_reward = 0.0
    done = False
    # At the beginning, we let it do at least one action. Because otherwise it can get stuck in a loop, where
    # 1) level L takes the current state and proposes subgoal A
    # 2) level L - 1 checks if the current state is close enough to A and it is
    # 3) we go back up to level L with the same unchanged state
    # 4) go back to (1)
    finished = False
    # if hac_params.is_top_level(level):
    #     finished = False
    # else:
    #     finished = num_attempts >= hac_params.max_horizons[level] or reached_any_supergoal(current_state, total_reward, subgoals_stack, level, hac_params)

    while not finished:
        # Step 1: sample a (noisy) action from the policy
        action, next_is_testing_subgoal = pick_action_and_testing(current_state, goal, level, is_testing_subgoal, env, hac_params, training)

        # Step 2: execute the action, either in the environment (if at the bottom level) or as a subgoal for the
        #         level below (if there's a level below)
        lower_level_layer_maxed_out = False  # If level > 0, this will be overriden by the real value
        if level > 0:
            # Train level i − 1 using subgoal ai
            subgoals_stack.append(action)
            next_state, action_reward, lower_level_layer_maxed_out, done = run_HAC_level(
                level - 1, current_state, action, env, hac_params, next_is_testing_subgoal, subgoals_stack, training, render
            )
            assert next_state is not None, "next_state is None!"
            subgoals_stack.pop()
        else:
            next_state, action_reward, done, _ = env.step(action)
            hac_params.step_number += 1

            next_state = next_state.astype(np.float32)
            if render:
                env_end_goal = np.array([0.0, 1.0, 0.0]) if env.spec.id == 'Pendulum-v0' else np.array([0.48, 0.04])
                if env.spec.id.startswith("Bipedal"):
                    env.render()
                elif hac_params.num_levels == 2:
                    env.unwrapped.render_goal(subgoals_stack[0][:-1], env_end_goal)
                elif hac_params.num_levels == 3:
                    env.unwrapped.render_goal_2(subgoals_stack[1][:-1], subgoals_stack[0][:-1], env_end_goal)

        # For debugging, log the Q-values
        if hac_params.use_tensorboard:
            if random.random() < 0.02:  # Don't log too often to avoid slowing things down
                value1 = hac_params.policies[level].critic1.forward(current_state, goal, action)
                value2 = hac_params.policies[level].critic2.forward(current_state, goal, action)
                value1_target = hac_params.policies[level].critic1_target.forward(current_state, goal, action)
                value2_target = hac_params.policies[level].critic2_target.forward(current_state, goal, action)

                writer, step_number = hac_params.writer, hac_params.step_number
                if level == 0:
                    for action_index in range(action.shape[0]):
                        writer.add_scalar(f"Action/{action_index} (Level {level})", action[action_index], step_number)
                else:
                    writer.add_scalar(f"Action/Predicted reward (Level {level})", action[-1], step_number)

                writer.add_scalar(f"Q-values/Normal Network 1 (Level {level})", value1, step_number)
                writer.add_scalar(f"Q-values/Normal Network 2 (Level {level})", value2, step_number)
                writer.add_scalar(f"Q-values/Target Network 1 (Level {level})", value1_target, step_number)
                writer.add_scalar(f"Q-values/Target Network 2 (Level {level})", value2_target, step_number)
                # writer.add_scalar("Action/Log prob action", log_prob, step_number)

        total_reward += action_reward
        if hac_params.use_tensorboard:
            hac_params.writer.add_scalar(f"Rewards/Action reward (level {level})", action_reward, hac_params.step_number)

        # Step 3: create replay transitions
        if level > 0 and lower_level_layer_maxed_out:
            if training and next_is_testing_subgoal:  # Penalize subgoal ai
                # Step 3a) Create "subgoal testing transition"
                # We want to penalize the lower level agent if it didn't reach the subgoal set by this agent
                did_reach_subgoal = reached_subgoal(next_state, total_reward, goal=action, level=level - 1, hac_params=hac_params)

                # "We use a discount rate of 0 in these transitions to avoid non-stationary transition function issues"
                # Case 1) Top level: We only add a subgoal penalty if we fail to reach the subgoal
                # If we reach the subgoal, we don't create a subgoal testing transition
                if hac_params.is_top_level(level) and not did_reach_subgoal:
                    testing_tr = mk_transition(current_state, action, action_reward, total_reward, next_state, HUGE_PENALTY, goal, discount=0)
                    hac_params.policies[level].add_to_buffer(testing_tr)

                # Case 2) Not top level: we add a subgoal transition in both cases (reach the subgoal or not) with different
                # penalties per case
                if not hac_params.is_top_level(level):
                    penalty = 0 if did_reach_subgoal else -hac_params.max_horizons[level]
                    testing_tr = mk_transition(current_state, action, action_reward, total_reward, next_state, penalty, goal, discount=0)
                    hac_params.policies[level].add_to_buffer(testing_tr)

            hindsight_action = np.hstack([next_state, action_reward])  # Replace original action with action executed in hindsight
        else:
            # There are 2 exception for the action hindsights where we don't replace the subgoal by the next state:
            # 1) If we reached the subgoal, then we can use the action (= subgoal)
            # 2) If we're at the lowest level, the action isn't a subgoal so it doesn't make sense to change it (the logic doesn't apply)
            hindsight_action = action

        # Step 3b) Create a "hindsight action transition"
        if training:
            # Evaluate executed action on current goal and hindsight goals
            # Here, compute_reward_and_discount only looks at the goal directly above (and not the layer upwards)
            # The paper isn't precise about this, but I checked the original code and they do the same
            # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L145
            action_tr_reward, action_tr_discount = compute_transition_reward_and_discount(next_state, action_reward, goal, level, done, hac_params)
            action_transition = mk_transition(
                current_state, hindsight_action, action_reward, total_reward, next_state, action_tr_reward, goal, action_tr_discount
            )
            hac_params.policies[level].add_to_buffer(action_transition)

        # Step 3c) Prepare the "hindsight goal transition"
        if training and not hac_params.is_top_level(level):  # There's no goal for the top level, so no need to do this
            # hindsight goal transitions would be created in the same way for the high level of the toy robot, except that the hindsight
            # goal transitions would be made from copies of the hindsight action transitions. Assuming
            # They need to be a list because they will be completed later on, and tuples don't allow modification in place
            # Things that are missing and will be completed are the transition reward, the goal, and the transition discount
            #                                                                                            TBD   TBD   TBD
            goal_transition = [current_state, hindsight_action, action_reward, total_reward, next_state, None, None, None]
            hac_params.her_storage[level].append(goal_transition)

        num_attempts += 1
        current_state = next_state

        if hac_params.is_top_level(level):
            finished = done
        else:
            finished = done or num_attempts >= hac_params.max_horizons[level] or \
                       reached_any_supergoal(current_state, total_reward, subgoals_stack, level, hac_params)

    # Step 3d (done when the action loop is completed): complete the "hindisght goal transition" using HER
    # and then add the completed transitions to the replay buffer
    if training and not hac_params.is_top_level(level):  # There's no goal for the top level, so no need to do this
        completed_goal_transitions = perform_HER(hac_params.her_storage[level], level, hac_params)
        hac_params.policies[level].add_many_to_buffer(completed_goal_transitions)
        hac_params.her_storage[level].clear()

    # Q: do we consider that reaching done means we maxed out?
    # Well, to answer that, we first need to understand what the purpose of maxed out is...
    # I don't remember, I'll look into it again.
    # Maxed out means we were unable to reach the goal. If it's d
    # There's 3 possibilities if we're finished:
    # 1) The environment is done
    # 2) We did H actions (only for non-top level)
    # 3) We reacher a super goal (only for non-top level)
    #
    # Top level: Maxed out only matters for the level above, so we don't have to care if we're the top level.
    # There's no maximum amount of steps for the top level anyway, so it would make any sense anyway to create a value for maxed_out
    #
    # Middle and bottom levels: the normal case can remain, but we also have to think about done. If done = True, should we
    # set maxed_out to True, or should we ignore the value of done? Well, it depends. In some environments, we want to reach done
    # so it's a good thing. In others, like Lundar Lander, done can be good (landed) or bad (crashed) so we can't know from that alone.
    # So it might be good or it might be bad, we don't know.
    # It might be dangerous to punish not reaching the subgoal because the environment is done. For example, if we manage to land
    # the lander, the environment will end. If we set maxed_out to True, then it will learn not to pick the subgoal that led to a good
    # landing. This is very bad. On the other hand, if there's a crash, we won't punish reaching a crash, but that should still be fine
    # because it will get lower rewards just through the environment.
    # Conclusion:
    # 1) If done is False, we keep the normal logic.
    # 2) If done is True, we ignore it and use the normal logic.
    if hac_params.is_top_level(level):
        maxed_out = None
    else:
        maxed_out = (num_attempts == hac_params.max_horizons[level] and
                     not reached_any_supergoal(current_state, total_reward, subgoals_stack, level, hac_params))

    return current_state, total_reward, maxed_out, done


def update_networks(hac_params: HacParams):
    for policy in hac_params.policies:
        policy.learn(hac_params.num_update_steps_when_training)


def run_hac(hac_params: HacParams, start_state: np.ndarray, goal_state: Optional[np.ndarray], env: gym.Env, training: bool, render: bool):
    return run_HAC_level(hac_params.num_levels - 1, start_state, goal_state, env, hac_params,
                         is_testing_subgoal=False, subgoals_stack=[], training=training, render=render)


def evaluate_hac(hac_params: HacParams, env: gym.Env, render_rounds: int, num_evals=20) -> Tuple[int, float, np.ndarray]:
    rewards = []
    with torch.no_grad():
        num_successes = 0
        for i in range(num_evals):
            state = env.reset()
            render_now = (i < render_rounds)
            _, reward, maxed_out, done = run_hac(hac_params, state, goal_state=None, env=env, training=False, render=render_now)

            if reward > hac_params.env_threshold:
                num_successes += 1
            rewards.append(reward)

    success_rate = num_successes / float(num_evals)
    return num_successes, success_rate, np.array(rewards)


def train(hac_params: HacParams, env: gym.Env, render_rounds: int, directory: str):
    running_reward = 0
    for i in tqdm(range(hac_params.num_training_episodes)):
        # Train General-HAC
        state = env.reset()
        _, episode_reward, _, done = run_hac(hac_params, state, goal_state=None, env=env, training=True, render=False)
        update_networks(hac_params)

        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        if hac_params.use_tensorboard:
            hac_params.writer.add_scalar(f"Rewards/Episode reward", episode_reward, i)
            hac_params.writer.add_scalar(f"Rewards/Running reward", running_reward, i)

        # Evaluate General-HAC
        if i == 0 or (i + 1) % hac_params.evaluation_frequency == 0:
            num_successes, success_rate, rewards = evaluate_hac(hac_params, env, render_rounds=render_rounds)
            print("\nStep %d: Success rate (%d/20): %.3f" % (i + 1, num_successes, success_rate))
            # noinspection PyStringFormat
            print("Reward: %.3f +- %.3f" % (np.mean(rewards), np.std(rewards)))

            if hac_params.use_tensorboard:
                hac_params.writer.add_scalar(f"Eval/Success rate", success_rate, i)
                hac_params.writer.add_scalar(f"Eval/Mean reward", np.mean(rewards), i)

            if success_rate == 1.0:
                print("Perfect success rate. Stopping training and saving model.")
                save_hac(hac_params, directory)
                return

        # Save General-HAC policies and params
        if (i + 1) % hac_params.save_frequency == 0:
            save_hac(hac_params, directory)


def save_hac(hac_params: HacParams, directory="."):
    # Create directory if it doesn't exit
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Save the policies at all levels
    policies_state_dicts = {f"policy_level_{i}": hac_params.policies[i].state_dict() for i in range(hac_params.num_levels)}
    torch.save(policies_state_dicts, f"{directory}/policies.ckpt")

    # Remove stuff we don't want to save
    policies_backup = hac_params.policies
    hac_params.policies = ["The models are stored in the 'policies.ckpt' file because otherwise this JSON file would be huge and unreadable."
                           "\n The load_hac() will deserialize both this JSON file and the policies, and then merge the results."]
    writer_backup = hac_params.writer
    hac_params.writer = None

    # Save the HAC parameters (without the agents and the buffers)
    with open(f'{directory}/hac_params.json', 'w') as f:
        json.dump(hac_params, f, default=json_default, indent=4, sort_keys=True)

    # Re-add the stuff we didn't want to save
    hac_params.policies = policies_backup
    hac_params.writer = writer_backup


def load_hac(directory: str = ".") -> HacParams:
    # Load the Hac Params
    with open(f'{directory}/hac_params.json', 'r') as f:
        hac_params_as_dict: dict = json.load(f)
        # Convert the lists into numpy arrays
        # Loop over a view to allow modification of the original dict as we iterate over it
        for key, value in hac_params_as_dict.items():
            if type(value) == list and key not in ["policies", "her_storage"]:
                hac_params_as_dict[key] = np.array(value, dtype=np.float32)

        hac_params = HacParams(**hac_params_as_dict)

    # Load the policies
    saved_policy_dicts = torch.load(f"{directory}/policies.ckpt")
    for level in range(hac_params.num_levels):
        policy_state_dict = saved_policy_dicts[f"policy_level_{level}"]
        hac_params.policies[level].load_state_dict(policy_state_dict)
        hac_params.policies[level].eval()

    return hac_params


