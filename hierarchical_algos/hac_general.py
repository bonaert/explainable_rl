import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union, Any
from pathlib import Path

import gym
import torch
import sklearn.preprocessing

from common import get_range_and_center, json_default, get_plan
from ddpg import DDPG
from sac import Sac, SacActor
from nicetypes import *


import os

from sacEntropyAdjustment import SacEntropyAdjustment


def scale_state(scaler: sklearn.preprocessing.StandardScaler, state: np.ndarray) -> np.ndarray:
    """ Scales the state given the scaler """
    return scaler.transform(state.reshape(1, -1))[0]


HUGE_PENALTY = -1000

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


# The question is: do the layer below get Q-values in the range [-H, 0] as in normal HAC or do we keep track of the real rewards
# at every layer? I don't think it's strictly necessary to use the real rewards, because we're trying to predict the real reward.
# Actors will try to pick actions that maximize the Q-values, so these Q-values have to reward accuracy for the non-top level layers.
# As long as failing to hit it leads to bad reward and hitting the subgoal lead to rewards, we should be good. So I think for
# subgoal-reaching layer we can keep [-H, 0]. But would using the real reward better? This would mean that reaching a subgoal
# with a higher reward would be incentivized than reaching a subgoal with a lower reward. However, not reaching the subgoal would be
# equally punished for both.

# Goals and subgoals are always pairs (desired state, desired reward)
# On the other hand, the action isn't always a subgoal, sadly...
# It would be nice if the goals and subgoals are always pairs
# On the other hand, the action can either be:
# 1) a low-level action
# 2) a subgoal (desired state, desired reward)
# Again, I think encoding this a single thing would be easier, and then we'd check the level we're at to see how to
# handle the action details when it's needed
# And should a subgoal be a tuple or numpy array? I think it's probably easier if it's a numpy array

# TODO(problem): There's a problem when I try the 3 level agent: it starts picking subgoals that are immediately reached.
# TODO(problem)  I probably need to add transitions that punish this kind of behavior.


def mk_transition(input: NumpyArray, action: NumpyArray, env_reward: float, total_env_reward: float,
                  next_input: NumpyArray, transition_reward: float, goal: NumpyArray, discount: float):
    """ This function is here just for type-checking and making it easier to ensure the order is correct through the IDE help """
    # Env reward: the reward given by the environment to the agent because it did the action
    # Total env reward: the sum of the rewards given by the environment after all the actions the agent has done
    # Transition reward: the reward we chose for this transition, which may be different than the reward given by the environment
    #        0       1         2             3              4              5            6       7
    return input, action, env_reward, total_env_reward, next_input, transition_reward, goal, discount


@dataclass
class HacParams:
    action_low: NumpyArray
    action_high: NumpyArray

    state_low: NumpyArray
    state_high: NumpyArray

    reward_low: List[float]
    reward_high: List[float]

    batch_size: int
    num_training_episodes: int
    num_levels: int
    max_horizons: List[int]
    discount: float
    replay_buffer_size: int
    subgoal_testing_frequency: float
    # If the distance between the state and the goal is below this threshold, we consider that we reached the goal
    state_distance_thresholds: List[List[float]]


    num_update_steps_when_training: int  # After the episode ended, we do X iterations of the optimization loop

    evaluation_frequency: int  # We evaluate the agent every X episodes

    save_frequency: int  # We save the model every X episodes

    env_threshold: float  # If the running reward is above this threshold, we solved the environment

    env_name: str  # Name of the OpenAI gym environment

    use_priority_replay: bool  # If True, use a Prioritized Replay Buffer instead of a normal Buffer

    penalty_subgoal_reachability: float  # The amount of punishment in the subgoal testing transition when failing to reach the goal

    q_bound_low_list: List[float]  # For each level, the lowest possible Q-value
    q_bound_high_list: List[float]  # For each level, the highest possible Q-value

    # (Optional) Teacher that can help the hierarchy learn
    # The teacher is used both to learn goal and low level actions, using rollouts of its actions
    teacher: Optional[SacActor] = None
    state_scaler: sklearn.preprocessing.StandardScaler = None
    probability_to_use_teacher: float = 0.5
    learn_low_level_transitions_from_teacher: bool = True

    # These fields have a default value but the user should be able
    # to override them.
    use_sac: bool = False  # By default we use DDPG, but we can switch to SAC
    use_sac_with_entropy_adjustment: bool = False
    alpha: float = 0.01
    all_levels_maximize_reward: bool = False  # If True, all levels maximize the reward while trying to reach the goal.
                                              # If False, they just try to reach the goal
    reward_present_in_input: bool = False   # Should the total environment reward be present in the input (could help with goal prediction)
    use_tensorboard: bool = True
    step_number: int = 0
    num_test_episodes: int = 10  # When evaluating the agent, how many episodes should we do?
    stop_episode_on_subgoal_failure: bool = False  # TODO(explain): why did I do this?
    use_reward_close_instead_of_above_minimum: bool = False  # If True, the obtained reward must be close to the predicted reward
                                                             # If False, the obtained reward must be greater than the desired minimum reward
    desired_reward_closeness: float = 0.5  # If the reward must be close instead of simply above the goal's reward,
                                           # this is the maximum distance allowed

    learning_rates: Optional[List[float]] = None  # Learning rate for each agent in the hierarchy

    # When adding Normal noise to the actions, we multiply the (high - low) / 2 by these
    # coefficients to ge the desired standard deviation
    # These are used in DDPG, but not in SAC, since SAC introduces noise natively through its entropy maximization
    # scheme and the use of a Normal Gaussian from which actions are sampled
    action_noise_coeffs: NumpyArray = None
    state_noise_coeffs: NumpyArray = None
    reward_noise_coeff: float = 0

    run_on_cluster: bool = False
    random_id: Optional[str] = 0
    data_dir_path: str = None

    # Fields with default value that will be filled with a true value in the __post_init__ method
    # Important: The user shouldn't fill these themselves! The values will be overwritten.
    state_size: int = -1
    action_size: int = -1

    # These are going to be overwritten anyway, so I can just initialize them to None
    action_range: NumpyArray = field(default_factory=lambda: None)
    action_center: NumpyArray = field(default_factory=lambda: None)

    subgoal_spaces_low: List[NumpyArray] = field(default_factory=list)
    subgoal_spaces_high: List[NumpyArray] = field(default_factory=list)
    subgoal_ranges: List[NumpyArray] = field(default_factory=list)
    subgoal_centers: List[NumpyArray] = field(default_factory=list)
    subgoal_noises_coeffs: List[NumpyArray] = field(default_factory=list)

    her_storage: List[List[list]] = field(default_factory=list)
    policies: List[Union[Sac, DDPG]] = field(default_factory=list)

    penalty_failure_reach_goal: List[float] = field(default_factory=list)

    writer: Any = field(default_factory=lambda: None)  # Tensorboard writer

    def __post_init__(self):
        # This method is executed at the end of the constructor.
        # I do some validation then setup some variables with their real value
        # This is useful for the user, because they don't have to it themselves and saves work
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
        assert self.learning_rates is None or len(self.learning_rates) == self.num_levels, "Error: incorrect number of learning rates"

        assert 0 <= self.probability_to_use_teacher <= 1, "Probability of using teacher must be between 0 and 1"
        assert not (self.teacher is not None and self.num_levels != 2), "Only support teacher for 2 level hierarchy"

        assert len(self.q_bound_low_list) == self.num_levels, "There must be as many Q bounds low as levels"
        assert len(self.q_bound_high_list) == self.num_levels, "There must be as many Q bounds high as levels"

        assert len(self.reward_low) == self.num_levels, \
            f"Reward_low has {len(self.reward_low)} values but there are {self.num_levels} levels"
        assert len(self.reward_high) == self.num_levels, \
            f"Reward_high has {len(self.reward_high)} values but there are {self.num_levels} levels"
        self.state_size = len(self.state_low)
        self.input_size = self.state_size + 1 if self.reward_present_in_input else self.state_size  # (State + reward) or State
        self.action_size = len(self.action_low)
        self.subgoal_size = self.state_size + 1  # (State + reward)

        # Only used in DDPG
        self.subgoal_noise_coeffs = None if self.state_noise_coeffs is None else np.hstack([self.state_noise_coeffs, self.reward_noise_coeff])

        for i in range(self.num_levels - 1):
            assert len(self.state_distance_thresholds[i]) == self.state_size, \
                "Number of distances thresholds at level %d is %d but should be %d (state dim)" % (
                    i, len(self.state_distance_thresholds[i]), self.state_size)

        assert self.subgoal_noise_coeffs is None or len(self.subgoal_noise_coeffs) == self.subgoal_size, \
            "Subgoal noise has %d dims but the states have %d dims" % (len(self.subgoal_noise_coeffs), self.subgoal_size)
        assert self.action_noise_coeffs is None or len(self.action_noise_coeffs) == self.action_size, \
            "Action noise has %d dims but the actions have %d dims" % (len(self.action_noise_coeffs), self.action_size)

        self.data_dir_path = os.environ['VSC_SCRATCH'] if self.run_on_cluster else '.'

        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer_id = datetime.now().strftime('%b%d_%H-%M-%S')
            if self.run_on_cluster:
                writer_id = writer_id + '-' + self.random_id

            self.writer = SummaryWriter(f"{self.data_dir_path}/logs/{self.env_name}/{writer_id}")

        self.her_storage = [[] for _ in range(self.num_levels)]
        self.policies = []

        self.action_range, self.action_center = get_range_and_center(self.action_low, self.action_high)

        # These Nones will be replaced by actual values
        self.subgoal_spaces_low, self.subgoal_spaces_high = [None] * self.num_levels, [None] * self.num_levels
        self.subgoal_centers, self.subgoal_ranges = [None] * self.num_levels, [None] * self.num_levels
        for level in range(self.num_levels):
            # Combine the state ranges and the reward ranges to create the subgoal ranges for each level
            if level > 0:
                self.subgoal_spaces_low[level] = np.hstack([self.state_low, self.reward_low[level]])
                self.subgoal_spaces_high[level] = np.hstack([self.state_high, self.reward_high[level]])
                self.subgoal_ranges[level], self.subgoal_centers[level] = get_range_and_center(
                    self.subgoal_spaces_low[level], self.subgoal_spaces_high[level]
                )

            q_bound_low = self.q_bound_low_list[level]  # None if (self.is_top_level(level) or self.all_levels_maximize_reward) else -self.max_horizons[level]
            q_bound_high = self.q_bound_high_list[level]  # None if (self.is_top_level(level) or self.all_levels_maximize_reward) else -self.max_horizons[level]
            learning_rate = 3e-4 if self.learning_rates is None else self.learning_rates[level]
            if self.use_sac:
                if self.use_sac_with_entropy_adjustment:
                    agent = SacEntropyAdjustment(
                        state_size=self.state_size if self.is_top_level(level) else self.input_size,
                        goal_size=self.subgoal_size if not self.is_top_level(level) else 0,
                        action_low=self.subgoal_spaces_low[level] if level > 0 else self.action_low,
                        action_high=self.subgoal_spaces_high[level] if level > 0 else self.action_high,
                        q_bound_low=q_bound_low,
                        q_bound_high=q_bound_high,
                        buffer_size=self.replay_buffer_size,
                        batch_size=self.batch_size,
                        writer=self.get_tensorboard_writer() if (self.use_tensorboard and not self.run_on_cluster) else None,
                        sac_id='Level %d' % level,
                        use_priority_replay=self.use_priority_replay,
                        learning_rate=learning_rate,
                        initial_alpha=self.alpha
                    )
                else:
                    agent = Sac(
                        state_size=self.state_size if self.is_top_level(level) else self.input_size,
                        goal_size=self.subgoal_size if not self.is_top_level(level) else 0,
                        action_low=self.subgoal_spaces_low[level] if level > 0 else self.action_low,
                        action_high=self.subgoal_spaces_high[level] if level > 0 else self.action_high,
                        q_bound_low=q_bound_low,
                        q_bound_high=q_bound_high,
                        buffer_size=self.replay_buffer_size,
                        batch_size=self.batch_size,
                        writer=self.get_tensorboard_writer() if (self.use_tensorboard and not self.run_on_cluster) else None,
                        sac_id='Level %d' % level,
                        use_priority_replay=self.use_priority_replay,
                        learning_rate=learning_rate,
                        alpha=self.alpha
                    )
            else:
                agent = DDPG(
                    state_size=self.state_size if self.is_top_level(level) else self.input_size,
                    goal_size=self.subgoal_size if level != self.num_levels - 1 else 0,  # No goal for the top level
                    action_range=self.subgoal_ranges[level] if level > 0 else self.action_range,
                    action_center=self.subgoal_centers[level] if level > 0 else self.action_center,
                    q_bound=q_bound_low,
                    buffer_size=self.replay_buffer_size,
                    batch_size=self.batch_size
                )
            self.policies.append(agent)

        # Create the "penalty_failure_reach_goal" for the case when we try to maximize the reward for the non-top-level policies
        for level in range(self.num_levels - 1):
            min_reward, max_reward = self.reward_low[level + 1], self.reward_high[level + 1]
            assert max_reward >= min_reward, f"Maximum reward ({max_reward}) < Min reward ({min_reward}) at level {level}"
            if min_reward < 0:
                self.penalty_failure_reach_goal.append(min_reward)
            else:
                self.penalty_failure_reach_goal.append(-(max_reward - min_reward) / self.max_horizons[level])

    def is_top_level(self, level: int) -> bool:
        return level == self.num_levels - 1

    def get_tensorboard_writer(self):
        """ Returns a Tensorboard writer, which allows logging many types of information (scalars, images, ...)"""
        return self.writer

    def should_use_teacher(self):
        assert self.num_levels == 2, "Teacher-guiding currently only implemented for hierarchy with 2 levels"
        return self.teacher is not None and random.random() < self.probability_to_use_teacher


def reached_subgoal(state: NumpyArray, cumulated_reward: float, goal: NumpyArray, level: int, hac_params: HacParams) -> bool:
    """ Reached the goal if the state is close enough to the goal, and if the cumulated reward is either
    above or close enough (depending on a flag) to the desired reward """
    return reached_desired_state(state, goal, level, hac_params) and reached_desired_reward(cumulated_reward, goal, hac_params)


def reached_desired_state(state: NumpyArray, goal: NumpyArray, level: int, hac_params: HacParams) -> bool:
    """ Reached the goal if the state is close enough to the goal  """
    close_enough_per_dim = is_close_enough_per_dim(state, goal, level, hac_params)
    return close_enough_per_dim.all()


def is_close_enough_per_dim(state: NumpyArray, goal: NumpyArray, level: int, hac_params: HacParams) -> NumpyArray:
    desired_state = goal[:-1]
    distances = np.abs(state - desired_state)
    return distances < hac_params.state_distance_thresholds[level]


def reached_desired_reward(cumulated_reward: float, goal: NumpyArray, hac_params: HacParams) -> bool:
    """ Reach the desired reward if the cumulated reward is either above or close enough (depending on a flag) to the desired reward """
    desired_reward = goal[-1]
    if hac_params.use_reward_close_instead_of_above_minimum:
        return abs(cumulated_reward - desired_reward) <= hac_params.desired_reward_closeness
    else:
        return cumulated_reward >= desired_reward


def reached_any_supergoal(current_state: NumpyArray, cumulated_reward: float, subgoals_stack: List[NumpyArray], level: int, hac_params: HacParams):
    for subgoal in subgoals_stack:
        if reached_subgoal(current_state, cumulated_reward, subgoal, level, hac_params):
            return True

    return False


def compute_transition_reward_and_discount(next_state: NumpyArray, action_reward: Optional[float], total_env_reward: float,
                                           goal: NumpyArray, subgoals_stack: List[NumpyArray], level: int, done: bool,
                                           is_last_step: bool, hac_params: HacParams) -> Tuple[float, float]:
    """
    If top level, use the classical reward scheme (reward=reward, discount=discount if not done else 0)
    For the other levels:
    - If not maximizing reward, use (reward=0, discount=0) or (reward=-1, discount=discount) reward scheme
    - If maximizing reward, use (reward=failure_reach_goal, discount=discount if not done else 0) or
                                (reward=total_env_reward, discount=0) in the success case
    """
    if hac_params.is_top_level(level):
        reward, discount = action_reward, (1.0 - float(done)) * hac_params.discount
    else:
        if hac_params.all_levels_maximize_reward:
            if reached_any_supergoal(next_state, total_env_reward, subgoals_stack, level, hac_params):
                reward, discount = total_env_reward, 0.0
            elif is_last_step:
                reward, discount = hac_params.penalty_failure_reach_goal[level], 0.0
            else:
                reward, discount = hac_params.penalty_failure_reach_goal[level], hac_params.discount
        else:
            if reached_subgoal(next_state, total_env_reward, goal, level, hac_params):
                reward, discount = 0.0, 0.0
            else:
                reward, discount = -1.0, hac_params.discount

    return reward, discount


def perform_HER(her_storage: List[list], level: int, subgoals_stack: List[NumpyArray], hac_params: HacParams) -> List[tuple]:
    """ Complete the hindsight goal transitions, by replacing the goal with one of the attained states in
    the transitions and then computing the rewards given the new goal """
    if len(her_storage) == 0:  # Can happen if we're directly at a subgoal
        return []

    transitions = her_storage[:]  # Make a copy to be sure we don't fuck things up
    completed_transitions = []

    # "First, one of the “next state” elements in one of the transitions will be selected
    #  as the new goal state replacing the TBD component in each transition"
    # random_transition_index = len(transitions) - 1 # random.randrange(int(len(transitions) * 0.75), len(transitions))
    random_transition_index = random.randrange(0, len(transitions))
    random_transition = transitions[random_transition_index]  # TODO(maybe use last one only): transitions[-1]
    total_env_reward, next_input = random_transition[3], random_transition[4]
    next_state = get_state_from_input(next_input, level, hac_params)
    chosen_env_reward = reduce_reward(total_env_reward, percentage=0.5)
    chosen_goal = np.hstack([next_state, chosen_env_reward])

    new_subgoals_stack = subgoals_stack[:]
    new_subgoals_stack[-1] = chosen_goal

    # The transitions that happened after the chosen goal are ignored (since they don't lead to the chosen goal,
    # instead happening afterwards)
    for i, transition in enumerate(transitions[:random_transition_index+1]):
    # for i, transition in enumerate(transitions):
        # We need to update the transition reward (5), the goal (6) and discount (7)
        # goal_transition = (current_state, action, env_reward, total_env_reward, next_state, None, None, None, done)
        tr_total_env_reward = transition[3]
        tr_next_state = get_state_from_input(transition[4], level, hac_params)
        # We use the TOTAL env reward, because we want to see if we reached the GOAL ABOVE
        # The done parameter won't be used, since we don't perform HER for the top level and the done parameter is used
        # only for the top level, so we arbitrarily set it to False
        reward, discount = compute_transition_reward_and_discount(
            tr_next_state, None, tr_total_env_reward, chosen_goal, new_subgoals_stack, level, done=False,
            is_last_step=(i == random_transition_index),
            # is_last_step=(i == len(transitions) - 1),
            hac_params=hac_params
        )
        transition[5] = reward
        transition[6] = chosen_goal
        transition[7] = discount

        completed_transitions.append(mk_transition(*transition))
    return completed_transitions


def get_action_low_and_high(env: gym.Env, hac_params: HacParams, level: int) -> Tuple[NumpyArray, NumpyArray]:
    if level == 0:
        return env.action_space.low, env.action_space.high
    else:
        return hac_params.subgoal_spaces_low[level], hac_params.subgoal_spaces_high[level]


def get_random_action(level: int, env: gym.Env, hac_params: HacParams) -> NumpyArray:
    return np.random.uniform(*get_action_low_and_high(env, hac_params, level))


def add_noise(action: NumpyArray, level: int, env: gym.Env, hac_params: HacParams) -> NumpyArray:
    noise_coeff = hac_params.action_noise_coeffs if level == 0 else hac_params.subgoal_noise_coeffs
    action += np.random.normal(0, noise_coeff)  # I'm using the Pytorch's implementation, it's different in the original TF one

    action_high, action_low = get_action_low_and_high(env, hac_params, level)
    action = np.clip(action, action_low, action_high)

    return action


def pick_action_and_testing(input: NumpyArray, goal: NumpyArray, level: int, is_testing_subgoal: bool,
                            env: gym.Env, hac_params: HacParams, training: bool) -> Tuple[NumpyArray, bool]:
    """ Returns the action (with maybe added noise) and whether we're go in testing mode """
    # If the layer above was testing, it requires that everything below it have deterministic (non-noisy)
    # behavior too. Therefore, this level must also be deterministic and be in "testing subgoal" mode
    # where we don't add any noise. Additionally, if we're not training but only evaluating the policy, we don't add noise.
    if is_testing_subgoal or not training:
        action = hac_params.policies[level].sample_action(input, goal, deterministic=True)
        return action, True

    # Exploration: Each level uses the following exploration strategy when a level is not involved in subgoal testing.
    # 10% of actions are sampled uniformly at random from the level’s action space
    # 90% of actions are the sum of actions sampled from the level’s policy and Gaussian noise
    if random.random() < 0.1 and hac_params.teacher is None:  # Don't use random actions if there's a teacher
        action = get_random_action(level, env, hac_params)
    else:
        if hac_params.use_sac:
            action = hac_params.policies[level].sample_action(input, goal, deterministic=False)
        else:
            action = hac_params.policies[level].sample_action(input, goal)
            action = add_noise(action, level, env, hac_params)

    # We start testing a certain fraction lambda of the time, e.g. with a probability lambda
    if random.random() < hac_params.subgoal_testing_frequency:
        is_testing_subgoal = True
    else:
        is_testing_subgoal = False

    return action, is_testing_subgoal


def reduce_reward(reward: float, percentage=0.1) -> float:
    """
    reduce_reward(100, 0.1)  =  100 - 100 * 0.1 =  100 - 10 =  90
    reduce_reward(-100, 0.1) = -100 - 100 * 0.1 = -100 - 10 = -110
    """
    return reward - abs(reward) * percentage


def expert_rollout(env: gym.Env, state: NumpyArray, hac_params: HacParams, training: bool) -> Tuple[NumpyArray, float, bool, int, List[Transition]]:
    # Step (1) do the rollout and collect incomplete transitions
    done = False
    total_reward = 0.0
    next_state = None
    incomplete_transitions = []

    # Decide the number of steps
    horizon_length = hac_params.max_horizons[0]
    num_steps = random.randint(int(0.75 * horizon_length), horizon_length)

    for i in range(num_steps):
        scaled_state = scale_state(hac_params.state_scaler, state) if hac_params.state_scaler is not None else state
        action = hac_params.teacher.get_actions(torch.tensor(scaled_state).float())
        next_state, reward, done, _ = env.step(action)

        total_reward += reward

        if training:
            incomplete_transitions.append((state, action, next_state, reward, total_reward, done))

        state = next_state

        if done:
            break

    # Step (2) create the low level transitions
    assert not hac_params.all_levels_maximize_reward, "Teacher-guiding currently not compatible with all levels maximizing reward"
    assert not hac_params.reward_present_in_input, "Teacher-guiding currently not compatible with reward present in input"

    final_state, reduced_total_reward = next_state, reduce_reward(total_reward, percentage=0.5)
    goal = np.hstack([final_state, reduced_total_reward])
    bottom_level_transitions = []
    for (state, action, next_state, reward, cumulated_reward, done) in incomplete_transitions:
        if reached_subgoal(state, cumulated_reward, goal, level=0, hac_params=hac_params):
            tr_reward, discount = 0.0, 0.0
        else:
            tr_reward, discount = -1.0, hac_params.discount

        # if done:
        #     discount = 0.0

        bottom_level_transitions.append(mk_transition(
            state, action, reward, cumulated_reward, next_state, tr_reward, goal, discount
        ))

    # Step (3) return all the relevant info
    lower_level_steps_taken = len(bottom_level_transitions)

    return final_state, total_reward, done, lower_level_steps_taken, bottom_level_transitions


# TODO(improvement): scaling the states seems to really improve the learning speed for
#                    LunarLander, maybe I should add that too
#                    some features have super small values and that might make it much harder
#                    to learn
def run_HAC_level(level: int, start_state: NumpyArray, goal: NumpyArray,
                  env: gym.Env, hac_params: HacParams,
                  level_above_is_testing_subgoal: bool, subgoals_stack: List[NumpyArray],
                  training: bool, render: bool) -> Tuple[NumpyArray, float, bool, bool, int, float]:
    """ Returns (last state, collected reward, if it failed to reach the goal, done)"""
    num_steps_taken = 0
    num_steps_could_reach_goal = 0
    total_num_steps = 0
    num_times_reached_subgoal = 0
    num_times_reached_state = 0
    num_times_reached_reward = 0
    num_times_reached_state_per_dim = None
    done = False

    current_state = start_state
    total_reward = 0.0
    current_input = build_input(current_state, total_reward, level, hac_params)

    # At the beginning, we let it do at least one action. Because otherwise it can get stuck in a loop, where
    # 1) level L takes the current state and proposes subgoal A
    # 2) level L - 1 checks if the current state is close enough to A and it is
    # 3) we go back up to level L with the same unchanged state
    # 4) go back to (1), repeating the loop
    finished = False
    while not finished:
        # Step 1: sample a (possibly noisy) action from the policy
        action, current_level_is_testing_subgoal = pick_action_and_testing(
            current_input, goal, level, level_above_is_testing_subgoal, env, hac_params, training
        )

        # Step 2: execute the action, either in the environment (if at the bottom level) or as a subgoal for the
        #         level below (if there's a level below)
        lower_level_failed_to_reach_its_goal = False  # If level > 0, this will be overriden by the real value
        use_teacher_to_pick_goal = False
        if level > 0:
            use_teacher_to_pick_goal = training and hac_params.should_use_teacher()  # Don't use teacher in testing
            if use_teacher_to_pick_goal:
                next_state, action_reward, done, lower_level_steps_taken, low_level_transitions = expert_rollout(env, current_state, hac_params, training)

                # Update variables to pretend we use HAC instead of teacher
                chosen_desired_reward = reduce_reward(action_reward, percentage=0.5)
                action = np.hstack([next_state, chosen_desired_reward])
                total_num_steps += lower_level_steps_taken

                # Add the transition for the lower level
                if hac_params.learn_low_level_transitions_from_teacher:
                    hac_params.policies[0].add_many_to_buffer(low_level_transitions)
            else:
                # Train level i − 1 using subgoal = "action we picked"
                subgoals_stack.append(action)
                next_state, action_reward, lower_level_failed_to_reach_its_goal, done, lower_level_steps_taken, _ = run_HAC_level(
                    level=level - 1,
                    start_state=current_state,
                    goal=action,
                    env=env,
                    hac_params=hac_params,
                    level_above_is_testing_subgoal=current_level_is_testing_subgoal,
                    subgoals_stack=subgoals_stack, training=training, render=render
                )
                subgoals_stack.pop()

                total_num_steps += lower_level_steps_taken
                num_steps_could_reach_goal += 1
                if not lower_level_failed_to_reach_its_goal:
                    num_times_reached_subgoal += 1
                if reached_desired_state(next_state, action, level - 1, hac_params):
                    num_times_reached_state += 1
                if reached_desired_reward(action_reward, action, hac_params):
                    num_times_reached_reward += 1
                
                if num_times_reached_state_per_dim is None:
                    num_times_reached_state_per_dim = is_close_enough_per_dim(next_state, action, level - 1, hac_params).astype(int)
                else:
                    num_times_reached_state_per_dim += is_close_enough_per_dim(next_state, action, level - 1, hac_params).astype(int)
                
        else:
            hac_params.step_number += 1

            next_state, action_reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)

            total_num_steps += 1

            if render:
                render_environment(env, hac_params, subgoals_stack, current_state)

        # Update total_reward and is_last_step
        total_reward += action_reward
        next_input = build_input(next_state, total_reward, level, hac_params)
        done = done or (hac_params.stop_episode_on_subgoal_failure and lower_level_failed_to_reach_its_goal)

        if hac_params.is_top_level(level):
            is_last_step = done
        else:
            is_last_step = (num_steps_taken == hac_params.max_horizons[level] - 1) or done

        # For debugging
        log_to_tensorboard(action, action_reward, current_input, goal, hac_params, level, next_state, total_reward)

        # Transition type (3) Subgoal testing transitions
        if training and current_level_is_testing_subgoal:
            if level > 0 and lower_level_failed_to_reach_its_goal and not use_teacher_to_pick_goal:
                # Step 3a) Create "subgoal testing transition"
                # We want to penalize the lower level agent if it didn't reach the subgoal set by this agent
                # "We use a discount rate of 0 in these transitions to avoid non-stationary transition function issues"
                if hac_params.is_top_level(level):
                    # TODO(check): I have to check if the subgoal penalties are any good
                    penalty = hac_params.penalty_subgoal_reachability
                elif not hac_params.is_top_level(level) and hac_params.all_levels_maximize_reward:
                    penalty = hac_params.penalty_subgoal_reachability
                else:  # not hac_params.is_top_level(level) and not hac_params.all_levels_maximize_reward:
                    penalty = -hac_params.max_horizons[level]

                testing_tr = mk_transition(current_input, action, action_reward, total_reward, next_input, penalty, goal, discount=0)
                hac_params.policies[level].add_to_buffer(testing_tr)

        # Create the 'hindsight action' by using the action if it's good, else replace it by the (state, reward)
        # the lower level actually reached
        if level > 0 and lower_level_failed_to_reach_its_goal:
            # TODO(think): can I switch to this without creating problems where the action I train on
            # TODO(think)  is different from the action the agent I actually picked
            chosen_reward = reduce_reward(action_reward, percentage=0.5)
            # chosen_reward = action_reward
            hindsight_action = np.hstack([next_state, chosen_reward])  # Replace original action with action executed in hindsight
        else:
            # There are 2 exception for the action hindsights where we don't replace the subgoal by the next state:
            # 1) If we reached the subgoal, then we can use the action (= subgoal)
            # 2) If we're at the lowest level, the action isn't a subgoal so it doesn't make sense to change it (the logic doesn't apply)
            hindsight_action = action

        # Transition type (1) Hindsight action transitions
        if training:
            # Evaluate executed action on current goal and hindsight goals
            # Here, compute_reward_and_discount only looks at the goal directly above (and not the layer upwards)
            # The paper isn't precise about this, but I checked the original code and they do the same
            # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L145
            # Total reward, because are comparing to the GOAL from ABOVE
            action_tr_reward, action_tr_discount = compute_transition_reward_and_discount(
                next_state, action_reward, total_reward, goal, subgoals_stack, level, done, is_last_step=is_last_step, hac_params=hac_params
            )
            action_transition = mk_transition(
                current_input, hindsight_action, action_reward, total_reward, next_input, action_tr_reward, goal, action_tr_discount
            )
            hac_params.policies[level].add_to_buffer(action_transition)

        # Transition type (2) Hindsight goal transitions (part 1/2: preparation)
        if training and not hac_params.is_top_level(level):  # There's no goal for the top level, so no need to do this
            # hindsight goal transitions would be created in the same way for the high level of the toy robot, except that the hindsight
            # goal transitions would be made from copies of the hindsight action transitions. Assuming
            # They need to be a list because they will be completed later on, and tuples don't allow modification in place
            # Things that are missing and will be completed are the transition reward, the goal, and the transition discount
            #                                                                                            TBD   TBD   TBD
            goal_transition = [current_input, hindsight_action, action_reward, total_reward, next_input, None, None, None]
            hac_params.her_storage[level].append(goal_transition)

        num_steps_taken += 1
        current_state = next_state
        current_input = next_input

        if hac_params.is_top_level(level):
            finished = done
        else:
            finished = done or num_steps_taken >= hac_params.max_horizons[level] or \
                       (reached_any_supergoal(current_state, total_reward, subgoals_stack, level, hac_params))

    # Transition type (2) Hindsight goal transitions (part 2/2: completion and addition to the replay buffer)
    # This must be done when the action loop is completed, since it requires having all transitions
    if training and not hac_params.is_top_level(level):  # There's no goal for the top level, so no need to do this
        completed_goal_transitions = perform_HER(hac_params.her_storage[level], level, subgoals_stack, hac_params)
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
        current_level_failed_to_reach_its_goal = Exception("Top level layer has no goal to be reached!")
    else:
        current_level_failed_to_reach_its_goal = (done or num_steps_taken == hac_params.max_horizons[level]) and \
                                                 not reached_any_supergoal(current_state, total_reward, subgoals_stack, level, hac_params)

    # Logs rewards, steps per episode, maxed out, done
    if hac_params.use_tensorboard:
        hac_params.writer.add_scalar(f"Level {level}/Total reward", total_reward, hac_params.step_number)
        hac_params.writer.add_scalar(f"Level {level}/Steps per episode", num_steps_taken, hac_params.step_number)
        if not hac_params.is_top_level(level):
            hac_params.writer.add_scalar(f"Level {level}/Failed to reach goal", current_level_failed_to_reach_its_goal, hac_params.step_number)

        hac_params.writer.add_scalar(f"Level {level}/Done", done, hac_params.step_number)

    if hac_params.is_top_level(level):
        print(f"Reached subgoals: {num_times_reached_subgoal}/{num_steps_could_reach_goal} ", end='')
        print(f"(failure state {num_steps_could_reach_goal - num_times_reached_state}/{num_steps_could_reach_goal}", end='')
        print(f" - {str(num_times_reached_state_per_dim)}/{num_steps_could_reach_goal}", end='')
        print(f", failure reward {num_steps_could_reach_goal - num_times_reached_reward}/{num_steps_could_reach_goal}", end='')
        print(f", expert: {num_steps_taken - num_steps_could_reach_goal}/{num_steps_taken})")
        if hac_params.use_tensorboard and num_steps_could_reach_goal > 0:
            hac_params.writer.add_scalar(f"Subgoals/Subgoal success", num_times_reached_subgoal / num_steps_could_reach_goal, hac_params.step_number)

        if num_steps_could_reach_goal == 0:
            percentage_reached_subgoal = 0.0
        else:
            percentage_reached_subgoal = num_times_reached_subgoal / num_steps_could_reach_goal
    else:
        percentage_reached_subgoal = -1.0

    return current_state, total_reward, current_level_failed_to_reach_its_goal, done, total_num_steps, percentage_reached_subgoal


def log_to_tensorboard(action, action_reward, current_input, goal, hac_params, level, next_state, total_reward):
    if hac_params.use_tensorboard:
        hac_params.writer.add_scalar(f"Rewards/Action reward (level {level})", action_reward, hac_params.step_number)
        hac_params.writer.add_scalar(f"Rewards/Total reward (level {level})", total_reward, hac_params.step_number)

        # if random.random() < 0.02:  # More extensive logging; don't log too often to avoid slowing things down
        #     value1 = hac_params.policies[level].critic1.forward(current_input, goal, action)
        #     value2 = hac_params.policies[level].critic2.forward(current_input, goal, action)
        #     value1_target = hac_params.policies[level].critic1_target.forward(current_input, goal, action)
        #     value2_target = hac_params.policies[level].critic2_target.forward(current_input, goal, action)
        #
        #     writer, step_number = hac_params.writer, hac_params.step_number
        #     if level == 0:
        #         for action_index in range(action.shape[0]):
        #             writer.add_scalar(f"Action/(Level {level}) {action_index} ", action[action_index], step_number)
        #
        #         for state_index in range(next_state.shape[0]):
        #             writer.add_scalar(f"States/(Level {level}) {state_index} ", next_state[state_index], step_number)
        #     else:
        #         writer.add_scalar(f"Action/Predicted reward (Level {level})", action[-1], step_number)
        #
        #     writer.add_scalar(f"Q-values/Normal (Level {level}) Network 1", value1, step_number)
        #     writer.add_scalar(f"Q-values/Normal (Level {level}) Network 2", value2, step_number)
        #     writer.add_scalar(f"Q-values/Target (Level {level}) Network 1", value1_target, step_number)
        #     writer.add_scalar(f"Q-values/Target (Level {level}) Network 2", value2_target, step_number)


def render_environment(env, hac_params, subgoals_stack, current_state):
    env_end_goal = np.array([0.0, 1.0, 0.0]) if env.spec.id == 'Pendulum-v0' else np.array([0.48, 0.04])
    if env.spec.id.startswith("Bipedal"):
        env.render()
    elif env.spec.id.startswith("Lunar"):
        if hac_params.num_levels == 2:
            plan_subgoals = get_plan(hac_params.policies[1].actor, current_state, num_iters=10, goal_has_reward=True)
        else:
            plan_subgoals = None
        env.unwrapped.render(state=current_state, goal=subgoals_stack[0][:-1], plan_subgoals=plan_subgoals)
    elif env.spec.id.startswith("MountainCar"):
        if hac_params.num_levels == 2:
            plan_subgoals = get_plan(hac_params.policies[1].actor, current_state, num_iters=4, goal_has_reward=True)
        else:
            plan_subgoals = None
        env.unwrapped.render(goal=subgoals_stack[0][:-1], end_goal=env_end_goal, plan_subgoals=plan_subgoals)
    elif hac_params.num_levels == 2:
        env.unwrapped.render_goal(subgoals_stack[0][:-1], env_end_goal)
    elif hac_params.num_levels == 3:
        env.unwrapped.render_goal_2(subgoals_stack[1][:-1], subgoals_stack[0][:-1], env_end_goal)


def build_input(state: NumpyArray, total_reward: float, level: int, hac_params: HacParams) -> NumpyArray:
    if hac_params.reward_present_in_input and not hac_params.is_top_level(level):
        # We don't put the reward into the state for the top level, because the top level doesn't
        # have to reach a certain minimum reward. Therefore, it has less of a need to receive the
        # reward as input and it may be better if we don't give it to the top level agent so that
        # it can better focus on the state (by ignoring the reward). This might allow faster and
        # better generalization.
        #
        # It's not clear in my head if it would hurt to include it, but since I'm not aware if it's
        # good, I will stick to the standard practice and not include it in my rewards.
        return np.hstack([state, total_reward])
    else:
        return state


def get_state_from_input(input: NumpyArray, level: int, hac_params: HacParams) -> NumpyArray:
    if hac_params.reward_present_in_input and not hac_params.is_top_level(level):
        return input[:-1]
    else:
        return input


def update_networks(hac_params: HacParams):
    for policy in hac_params.policies:
        policy.learn(hac_params.num_update_steps_when_training)


def run_hac(hac_params: HacParams, start_state: NumpyArray, goal_state: Optional[NumpyArray], env: gym.Env, training: bool, render: bool):
    return run_HAC_level(level=hac_params.num_levels - 1, start_state=start_state, goal=goal_state, env=env,
                         hac_params=hac_params, level_above_is_testing_subgoal=False, subgoals_stack=[],
                         training=training, render=render)


def evaluate_hac(hac_params: HacParams, env: gym.Env, render_rounds: int, num_evals: int) -> Tuple[int, float, float, NumpyArray, NumpyArray]:
    rewards = []
    num_steps_per_episode = []
    num_times_mostly_reached_subgoal = 0
    with torch.no_grad():
        num_successes = 0
        for i in range(num_evals):
            state = env.reset()
            render_now = (i < render_rounds)
            _, reward, maxed_out, done, total_num_steps, percentage_reached_subgoal = run_hac(hac_params, state, goal_state=None, env=env, training=False, render=render_now)

            if reward >= hac_params.env_threshold:
                num_successes += 1
            if percentage_reached_subgoal >= 0.8:
                num_times_mostly_reached_subgoal += 1

            rewards.append(reward)
            num_steps_per_episode.append(total_num_steps)

            print(f"Total reward: {reward}")

    success_rate = num_successes / float(num_evals)
    reached_subgoal_rate = num_times_mostly_reached_subgoal / float(num_evals)
    return num_successes, success_rate, reached_subgoal_rate, np.array(rewards), np.array(num_steps_per_episode),


def train(hac_params: HacParams, env: gym.Env, render_rounds: int, directory: str):
    running_reward = 0.0
    for i in range(hac_params.num_training_episodes):
        state = env.reset()
        _, episode_reward, _, done, num_episode_steps, _ = run_hac(
            hac_params, state, goal_state=None, env=env, training=True, render=False
        )
        running_reward = 0.05 * episode_reward + 0.95 * running_reward

        print(f"Ep {i}/{hac_params.num_training_episodes}\t"
              # f"Avg. reward: {float(episode_reward) / num_episode_steps:.2f}\t"
              f"Ep. reward: {float(episode_reward):.2f}\t"
              f"R. Reward: {running_reward:.2f}\t"
              f"Steps: {num_episode_steps}")

        update_networks(hac_params)

        if hac_params.use_tensorboard:
            hac_params.writer.add_scalar(f"Progress/Episode reward", episode_reward, i)
            hac_params.writer.add_scalar(f"Progress/Running reward", running_reward, i)
            hac_params.writer.add_scalar(f"Progress/Num episodes steps", num_episode_steps, i)

        # Evaluate General-HAC
        if i == 0 or (i + 1) % hac_params.evaluation_frequency == 0:
            num_successes, success_rate, reached_subgoal_rate, rewards, steps_per_episode = evaluate_hac(
                hac_params, env, render_rounds=render_rounds, num_evals=hac_params.num_test_episodes
            )
            print("\nStep %d: Success rate (%d/%d): %.3f" % (i + 1, num_successes, hac_params.num_test_episodes, success_rate))
            print("Reached subgoal rate: %.3f" % reached_subgoal_rate)
            # noinspection PyStringFormat
            print("Reward: %.3f +- %.3f" % (np.mean(rewards), np.std(rewards)))
            # noinspection PyStringFormat
            print("Number of steps: %.3f +- %.3f" % (np.mean(steps_per_episode), np.std(steps_per_episode)))

            if hac_params.use_tensorboard:
                hac_params.writer.add_scalar(f"Eval/Success rate", success_rate, i)
                hac_params.writer.add_scalar(f"Eval/Goal Success rate", reached_subgoal_rate, i)
                hac_params.writer.add_scalar(f"Eval/Mean reward", np.mean(rewards), i)
                hac_params.writer.add_scalar(f"Eval/Mean number of steps", np.mean(steps_per_episode), i)
                hac_params.writer.add_scalar(f"Eval/Std dev number of steps", np.std(steps_per_episode), i)

            if np.mean(rewards) > hac_params.env_threshold and reached_subgoal_rate > 0.6:
                # Evaluate over more episodes to be sure it's good enough
                _, _, reached_subgoal_rate, many_rewards, _ = evaluate_hac(hac_params, env, render_rounds=0, num_evals=20)

                if np.mean(many_rewards) > hac_params.env_threshold and reached_subgoal_rate > 0.6:
                    print("Perfect success rate. Stopping training and saving model.")
                    save_hac(hac_params, directory)
                    return

        # Save General-HAC policies and params
        if (i + 1) % hac_params.save_frequency == 0:
            save_hac(hac_params, directory)

        # Flush tensorboard at the end of each episode to ensure things get written to file
        if hac_params.use_tensorboard:
            hac_params.writer.flush()


def save_hac(hac_params: HacParams, directory="."):
    # Create directory if it doesn't exit
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Save the policies at all levels
    policies_state_dicts = {f"policy_level_{i}": hac_params.policies[i].state_dict() for i in range(hac_params.num_levels)}
    torch.save(policies_state_dicts, f"{directory}/policies.ckpt")

    # Remove stuff we don't want to save
    policies_backup = hac_params.policies
    writer_backup = hac_params.writer
    teacher_backup = hac_params.teacher
    scaler_backup = hac_params.state_scaler
    hac_params.policies = ["The models are stored in the 'policies.ckpt' file because otherwise this JSON file would be huge and unreadable."
                           "\n The load_hac() will deserialize both this JSON file and the policies, and then merge the results."]
    hac_params.writer = None
    hac_params.teacher = None
    hac_params.state_scaler = None

    # Save the HAC parameters (without the agents and the buffers)
    with open(f'{directory}/hac_params.json', 'w') as f:
        json.dump(hac_params, f, default=json_default, indent=4, sort_keys=True)

    # Re-add the stuff we didn't want to save
    hac_params.policies = policies_backup
    hac_params.writer = writer_backup
    hac_params.teacher = teacher_backup
    hac_params.state_scaler = scaler_backup


def load_hac(directory: str = ".") -> HacParams:
    # Load the Hac Params
    with open(f'{directory}/hac_params.json', 'r') as f:
        print(f"Loading from file {directory}/hac_params.json ")
        hac_params_as_dict: dict = json.load(f)
        # Convert the lists into numpy arrays
        # Loop over a view to allow modification of the original dict as we iterate over it
        for key, value in hac_params_as_dict.items():
            if type(value) == list and key not in ["policies", "her_storage",
                                                   "subgoal_spaces_low", "subgoal_spaces_high",
                                                   "subgoal_centers", "subgoal_ranges", "reward_low", "reward_high", "penalty_failure_reach_goal"]:
                hac_params_as_dict[key] = np.array(value, dtype=np.float32)

        hac_params_as_dict['run_on_cluster'] = False
        hac_params = HacParams(**hac_params_as_dict)

    # Load the policies
    saved_policy_dicts = torch.load(f"{directory}/policies.ckpt")
    for level in range(hac_params.num_levels):
        policy_state_dict = saved_policy_dicts[f"policy_level_{level}"]
        hac_params.policies[level].load_state_dict(policy_state_dict)
        hac_params.policies[level].eval()

    return hac_params
