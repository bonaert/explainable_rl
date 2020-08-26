import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from pathlib import Path
from datetime import datetime

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

from common import get_tensor, ReplayBuffer, get_range_and_center, json_default, NEVER, FIRST_RUN, ALWAYS

class Actor(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_size: int,
                 action_range: np.ndarray, action_center: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(state_size + goal_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.action_range = torch.from_numpy(action_range)
        self.action_center = torch.from_numpy(action_center)

    def forward(self, state: np.ndarray, goal: np.ndarray) -> torch.Tensor:
        state, goal = get_tensor(state), get_tensor(goal)
        total_input = torch.cat([state, goal], dim=-1)  # Concatenate to format [states | goals]
        x = self.fc1(total_input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return self.action_center + self.action_range * torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_size: int, q_bound: float):
        super().__init__()
        self.fc1 = nn.Linear(state_size + goal_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.q_bound = q_bound

        # self.q_init = -0.67  # This is -1/15... TODO: figure out why they picked this...
        # self.q_offset = -np.log(self.q_bound / self.q_init - 1)  # TODO: figure out why this formula is used

    def forward(self, state: np.ndarray, goal: np.ndarray, action: np.ndarray) -> torch.Tensor:
        state, goal, action = get_tensor(state), get_tensor(goal), get_tensor(action)
        total_input = torch.cat([state, goal, action], dim=-1)  # Concatenate to format [states | goals | actions]
        x = self.fc1(total_input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        # TODO: self.q_init is a value that leads to a good initialization by default, so that the
        # sigmoid is near 0. In this code I don't use it. See if it's needed
        return self.q_bound * torch.sigmoid(x)  # -5 * [0, 1] -> [-5, 0]


class DDPG(nn.Module):
    def __init__(self, state_size: int, goal_size: int, action_range: np.ndarray, action_center: np.ndarray,
                 q_bound: float, buffer_size: int, batch_size: int):
        super().__init__()
        action_size = len(action_range)

        # Important: there are no target networks on purpose, because the HAC paper
        # found they were not very useful
        self.critic = Critic(state_size, goal_size, action_size, q_bound)
        self.actor = Actor(state_size, goal_size, action_size, action_range, action_center)

        # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/master/critic.py#L8
        self.critic_optimizer = Adam(self.critic.parameters(), lr=0.001)
        # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/master/actor.py#L15
        self.actor_optimizer = Adam(self.actor.parameters(), lr=0.001)

        # There's 6 dimensions in a transition: (current_state, action, penalty, next_state, current_goal, discount)
        # NOTE: they use some more complicated logic (which depends on the level) to determinate the size of the buffer
        # TODO: this is a simplfication. See if it works anyway.
        self.buffer = ReplayBuffer(buffer_size, num_transition_dims=6)
        self.batch_size = batch_size
        self.q_bound = q_bound

    def add_to_buffer(self, transition: tuple):
        self.buffer.add(transition)

    def add_many_to_buffer(self, transitions: List[tuple]):
        self.buffer.add_many(transitions)

    def sample_action(self, state: np.ndarray, goal: np.ndarray):
        with torch.no_grad():
            return self.actor(state, goal).numpy()

    def learn(self, num_updates: int):
        # If there's not enough transitions to fill a batch, we don't do anything
        if self.buffer.size() < self.batch_size:
            return

        for i in range(num_updates):
            # Step 1: get the transitions and the next actions for the next state
            states, actions, rewards, next_states, goals, discounts = self.buffer.get_batch(self.batch_size)
            next_actions = self.actor(next_states, goals)

            # Step 2: improve the critic
            with torch.no_grad():  # No need to compute gradients for this
                target_q_values = rewards + discounts * self.critic(next_states, goals, next_actions).squeeze()
                target_q_values = target_q_values.unsqueeze(1)
                # We clamp the Q-values to be in [-H, 0]. Why would this be needed given that the critic already
                # outputs values in this range? Well, it's true, the critic does do that, but the target
                # expression is r + alpha * Q(s', a') and that might go outside of [-H, 0]. We don't want
                # that to happen, so we clamp it to the range. This will thus incentivize the critic to predict
                # values in [-H, 0], but since the critic can already only output values in that range, it's perfect.
                # Of course, this is not a coincidence but done by design.
                target_q_values = torch.clamp(target_q_values, min=self.q_bound, max=0.0)

            self.critic_optimizer.zero_grad()
            predicted_q_values = self.critic(states, goals, actions)
            critic_loss = F.mse_loss(predicted_q_values, target_q_values)
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 3: improve the actor
            # Freeze Q-network so you don't waste computational effort
            # computing gradients for it during the policy learning step.
            # TODO: for some reason, if I do this, then I get this error when I do actor_loss.backward()
            # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
            # This does not happen in my other DDPG code and I don't know why.
            # TODO: figure it out
            # for p in self.critic.parameters():
            #     p.requires_grad = False

            # We want to maximize the q-values of the actions (and therefore minimize -Q_values)
            self.actor_optimizer.zero_grad()
            new_actions = self.actor(states, goals)
            actor_loss = -self.critic(states, goals, new_actions).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze Q-network so you can optimize it at next DDPG step.
            # for p in self.critic.parameters():
            #     p.requires_grad = True


@dataclass
class HacParams:
    action_low: np.ndarray
    action_high: np.ndarray

    state_low: np.ndarray
    state_high: np.ndarray

    batch_size: int
    num_training_episodes: int
    num_levels: int
    max_horizons: List[int]
    discount: float
    replay_buffer_size: int
    subgoal_testing_frequency: float
    # If the distance between the state and the goal is below this threshold, we consider that we reached the goal
    distance_thresholds: List[List[float]]
    # When adding Normal noise to the actions, we multiply the (high - low) / 2 by these
    # coefficients to ge the desired standard deviation
    action_noise_coeffs: np.ndarray
    subgoal_noise_coeffs: np.ndarray

    num_update_steps_when_training: int

    evaluation_frequency: int

    save_frequency: int

    env_name: str

    run_on_cluster: bool = False
    random_id: Optional[str] = 0

    # Fields with default value that will be filled with a true value in the __post_init__ method
    state_size: int = -1
    action_size: int = -1

    # These are going to be overwritten anyway, so I can just initialize them to None
    action_range: np.ndarray = field(default_factory=lambda: None)
    action_center: np.ndarray = field(default_factory=lambda: None)
    state_range: np.ndarray = field(default_factory=lambda: None)
    state_center: np.ndarray = field(default_factory=lambda: None)

    her_storage: List[List[list]] = field(default_factory=list)
    policies: List[DDPG] = field(default_factory=list)

    def __post_init__(self):
        if self.env_name.startswith("LunarLander"):
            self.state_low = np.array([-2, -5, -3, -3, -5, -5, 0, 0], dtype=np.float32)
            self.state_high = np.array([2,  5,  3,  3,  5,  5, 1, 1], dtype=np.float32)

        # This method is executed at the end of the constructor. Here, I can setup the list I need
        # I do some validation then setup some variables with their real value
        # This is useful for the user, which doesn't have to it themselves and saves work
        # It also ensures it's done correctly
        assert 0 <= self.subgoal_testing_frequency <= 1, "Subgoal testing frequency must be between 0 and 1"
        assert 0 <= self.discount <= 1, "Discount must be between 0 and 1"
        assert 1 <= self.num_levels, "The number of levels must be at least 1"
        assert (1 <= np.array(self.max_horizons)).all(), "All horizons must at least be 1 step long"
        assert len(self.max_horizons) == self.num_levels, "There must be as many horizons as the number of levels"
        assert len(self.distance_thresholds) == self.num_levels, \
            "Number of distances thresholds (%d) is different from the number of levels (%d)" % (
                len(self.distance_thresholds), self.num_levels)
        assert not np.isinf(self.action_low).any(), "Error: the action space cannot have +-infinite lower bounds"
        assert not np.isinf(self.action_high).any(), "Error: the action space cannot have +-infinite upper bounds"
        assert not np.isinf(self.state_low).any(), "Error: the state space cannot have +-infinite lower bounds"
        assert not np.isinf(self.state_high).any(), "Error: the state space cannot have +-infinite upper bounds"

        self.action_range, self.action_center = get_range_and_center(self.action_low, self.action_high)
        self.state_range, self.state_center = get_range_and_center(self.state_low, self.state_high)

        self.state_size = len(self.state_low)
        self.action_size = len(self.action_low)

        for i in range(self.num_levels):
            assert len(self.distance_thresholds[i]) == self.state_size, \
                "Number of distances thresholds at level %d is %d but should be %d (state dim)" % (
                    i, len(self.distance_thresholds[i]), self.state_size)

        assert len(self.subgoal_noise_coeffs) == self.state_size, \
            "Subgoal noise has %d dims but the states have %d dims" % (len(self.subgoal_noise_coeffs), self.state_size)
        assert len(self.action_noise_coeffs) == self.action_size, \
            "Action noise has %d dims but the actions have %d dims" % (len(self.action_noise_coeffs), self.action_size)

        self.her_storage = [[] for _ in range(self.num_levels)]
        self.policies = []
        for level in range(self.num_levels):
            ddpg = DDPG(
                state_size=self.state_size,
                goal_size=self.state_size,
                action_range=self.state_range if level > 0 else self.action_range,
                action_center=self.state_center if level > 0 else self.action_center,
                q_bound=-self.max_horizons[level],
                buffer_size=self.replay_buffer_size,
                batch_size=self.batch_size
            )
            self.policies.append(ddpg)


def reached_subgoal(state: np.ndarray, goal: np.ndarray, level: int, hac_params: HacParams) -> bool:
    distances = np.abs(state - goal)
    return (distances < hac_params.distance_thresholds[level]).all()


def reached_any_supergoal(current_state: np.ndarray, subgoals_stack: List[np.ndarray], level: int, hac_params: HacParams):
    for subgoal in subgoals_stack:
        if reached_subgoal(current_state, subgoal, level, hac_params):
            return True

    return False


def compute_reward_and_discount(state: np.ndarray, goal: np.ndarray, level: int, hac_params: HacParams) -> Tuple[float, float]:
    if reached_subgoal(state, goal, level, hac_params):
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
    chosen_next_state = random.choice(transitions)[3]

    for transition in transitions:
        # We need to update the reward (2), the goal (4) and discount (5)
        #                                           TBD               TBD   TBD
        # goal_transition = (current_state, action, None, next_state, None, None)
        current_next_state = transition[3]
        reward, discount = compute_reward_and_discount(current_next_state, chosen_next_state, level, hac_params)
        transition[2] = reward
        transition[4] = chosen_next_state  # Goal
        transition[5] = discount

        completed_transitions.append(tuple(transition))

    return completed_transitions


def get_random_action(level: int, env: gym.Env) -> np.ndarray:
    if level == 0:
        return np.random.uniform(env.action_space.low, env.action_space.high)
    else:
        state_low, state_high = env.observation_space.low, env.observation_space.high
        if env.spec.id.startswith("LunarLander"):
            state_low = np.array([-2, -5, -3, -3, -5, -5, 0, 0], dtype=np.float32)
            state_high = np.array([2,  5,  3,  3,  5,  5, 1, 1], dtype=np.float32)
        return np.random.uniform(state_low, state_high)


def add_noise(action: np.ndarray, level: int, env: gym.Env, hac_params: HacParams) -> np.ndarray:
    if level == 0:
        action_low, action_high = env.action_space.low, env.action_space.high
    else:
        action_low, action_high = env.observation_space.low, env.observation_space.high

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
        action = hac_params.policies[level].sample_action(state, goal)
        return action, True

    # Exploration: Each level uses the following exploration strategy when a level is not involved in subgoal testing.
    # 20% of actions are sampled uniformly at random from the level’s action space
    # 80% of actions are the sum of actions sampled from the level’s policy and Gaussian noise
    if random.random() < 0.2:
        action = get_random_action(level, env)
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
                  training: bool, render: bool) -> Tuple[np.ndarray, bool, float, bool]:
    current_state = start_state
    num_attempts = 0
    total_reward = 0
    done = False
    while not done and num_attempts < hac_params.max_horizons[level] and not reached_any_supergoal(current_state, subgoals_stack, level, hac_params):
        # Step 1: sample a (noisy) action from the policy
        action, next_is_testing_subgoal = pick_action_and_testing(current_state, goal, level, is_testing_subgoal, env, hac_params, training)

        # Step 2: execute the action, either in the environment (if at the bottom level) or as a subgoal for the
        #         level below (if there's a level below)
        lower_level_layer_maxed_out = False  # If level > 0, this will be overriden by the real value
        if level > 0:
            # Train level i − 1 using subgoal ai
            subgoals_stack.append(action)
            next_state, lower_level_layer_maxed_out, reward, done = run_HAC_level(level - 1, current_state, action, env, hac_params,
                                                                    next_is_testing_subgoal, subgoals_stack, training, render)
            assert next_state is not None, "next_state is None!"
            subgoals_stack.pop()
        else:
            next_state, reward, done, _ = env.step(action)
            if render:
                env.unwrapped.render(goal=goal, end_goal=None, mode='human', plan_subgoals=None)
                # if hac_params.num_levels == 2:
                #     env.unwrapped.render_goal(*subgoals_stack[::-1])
                # elif hac_params.num_levels == 3:
                #     env.unwrapped.render_goal_2(*subgoals_stack[::-1])

        total_reward += reward

        # Step 3: create replay transitions
        if level > 0 and lower_level_layer_maxed_out:
            if training and next_is_testing_subgoal:  # Penalize subgoal ai
                # Step 3a) Create "subgoal testing transition"
                # We want to penalize the lower level agent if it didn't reach the subgoal set by this agent
                if reached_subgoal(next_state, goal=action, level=level - 1, hac_params=hac_params):  # The action is the su
                    penalty = 0  # We were able to reach the
                else:
                    penalty = -hac_params.max_horizons[level]

                # "We use a discount rate of 0 in these transitions to avoid non-stationary transition function issues"
                testing_transition = (current_state, action, penalty, next_state, goal, 0)
                hac_params.policies[level].add_to_buffer(testing_transition)

            hindsight_action = next_state  # Replace original action with action executed in hindsight
        else:
            # There are 2 exception for the action hindsights where we don't replace the subgoal by the next state:
            # 1) If we reached the subgoal, then we can use the action (= subgoal)
            # 2) If we're at the lowest level, the action isn't a subgoal so it doesn't make sense to change it (the logic doesn't apply)
            hindsight_action = action

        if training:
            # Evaluate executed action on current goal and hindsight goals
            # Step 3b) Create a "hindsight action transition"
            # Here, compute_reward_and_discount only looks at the goal directly above (and not the layer upwards)
            # The paper isn't precise about this, but I checked the original code and they do the same
            # https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-/blob/f90f2c356ab0a95a57003c4d70a0108f09b6e6b9/layer.py#L145
            action_transition_reward, action_transition_discount = compute_reward_and_discount(next_state, goal, level, hac_params)
            action_transition = (current_state, hindsight_action, action_transition_reward, next_state, goal, action_transition_discount)
            hac_params.policies[level].add_to_buffer(action_transition)

            # Step 3c) Prepare the "hindsight goal transition"
            # hindsight goal transitions would be created in the same way for the high level of the toy robot, except that the hindsight
            # goal transitions would be made from copies of the hindsight action transitions. Assuming
            # They need to be a list because they will be completed later on, and tuples don't allow modification in place
            #                                         TBD               TBD   TBD
            goal_transition = [current_state, hindsight_action, None, next_state, None, None]
            hac_params.her_storage[level].append(goal_transition)

        num_attempts += 1
        current_state = next_state

    if training:
        # Step 3d (done when the action loop is completed): complete the "hindisght goal transition" using HER
        # and then add the completed transitions to the replay buffer
        completed_goal_transitions = perform_HER(hac_params.her_storage[level], level, hac_params)
        hac_params.policies[level].add_many_to_buffer(completed_goal_transitions)
        hac_params.her_storage[level].clear()

    # Step 4: return the current (final) state and maxed_out
    maxed_out = done or (num_attempts == hac_params.max_horizons[level] and not reached_any_supergoal(current_state, subgoals_stack, level, hac_params))
    return current_state, maxed_out, total_reward, done


def update_networks(hac_params: HacParams):
    for policy in hac_params.policies:
        policy.learn(hac_params.num_update_steps_when_training)


def run_hac(hac_params: HacParams, start_state: np.ndarray, goal_state: np.ndarray, env: gym.Env, training: bool, render: bool):
    return run_HAC_level(hac_params.num_levels - 1, start_state, goal_state, env, hac_params,
                         is_testing_subgoal=False, subgoals_stack=[goal_state], training=training, render=render)


def evaluate_hac(hac_params: HacParams, env: gym.Env, goal_state: np.ndarray,
                 render_frequency: int, num_evals: int = 20) -> Tuple[int, float, float]:
    with torch.no_grad():
        num_successes = 0
        cumulated_reward = 0
        for i in range(num_evals):
            state = env.reset()
            render_now = (render_frequency == ALWAYS) or (render_frequency == FIRST_RUN and i == 0)
            _, failed, total_reward, done = run_hac(hac_params, state, goal_state, env, training=False, render=render_now)

            cumulated_reward += total_reward
            if not failed:
                num_successes += 1

    success_rate = num_successes / float(num_evals)
    average_reward = cumulated_reward / float(num_evals)
    return num_successes, success_rate, average_reward


def train(hac_params: HacParams, env: gym.Env, goal_state: np.ndarray, render_frequency: int, save_directory: str):
    writer_id = datetime.now().strftime('%b%d_%H-%M-%S')
    if hac_params.run_on_cluster:
        writer_id = writer_id + '-' + hac_params.random_id

    prefix = os.environ['VSC_SCRATCH'] if hac_params.run_on_cluster else '.'
    writer = SummaryWriter(f"{prefix}/logs/HAC/{env.spec.id}/{writer_id}")

    # for i in tqdm(range(hac_params.num_training_episodes)):
    for i in range(hac_params.num_training_episodes):
        # Train HAC
        state = env.reset()
        run_hac(hac_params, state, goal_state, env, training=True, render=False)
        update_networks(hac_params)

        # Evaluate HAC
        if i == 0 or (i + 1) % hac_params.evaluation_frequency == 0:
            num_successes, success_rate, avg_reward = evaluate_hac(hac_params, env, goal_state, render_frequency)
            print("\nStep %d: Success rate (%d/20): %.3f, Average Reward %.3f" % (i + 1, num_successes, success_rate, avg_reward))

            writer.add_scalar('Eval/Success Rate', success_rate, i)
            writer.add_scalar('Eval/Mean Reward', avg_reward, i)

            if success_rate == 1.0:
                print("Perfect success rate. Stopping training and saving model.")
                save_hac(hac_params, save_directory)
                return

        # Save HAC policies and params
        if (i + 1) % hac_params.save_frequency == 0:
            save_hac(hac_params, save_directory)


def save_hac(hac_params: HacParams, directory: str = "."):
    # Create directory if it doesn't exit
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Save the policies at all levels
    policies_state_dicts = {f"policy_level_{i}": hac_params.policies[i].state_dict() for i in range(hac_params.num_levels)}
    torch.save(policies_state_dicts, f"{directory}/policies.ckpt")

    # Save the HAC parameters (without the agents and the buffers)
    policies_backup = hac_params.policies
    hac_params.policies = ["The models are stored in the 'policies.ckpt' file because otherwise this JSON file would be huge and unreadable."
                           "\n The load_hac() will deserialize both this JSON file and the policies, and then merge the results."]
    with open(f'{directory}/hac_params.json', 'w') as f:
        json.dump(hac_params, f, default=json_default, indent=4, sort_keys=True)
    hac_params.policies = policies_backup


def load_hac(directory: str = ".") -> HacParams:
    # Load the Hac Params
    with open(f'{directory}/hac_params.json', 'r') as f:
        hac_params_as_dict: dict = json.load(f)
        # Convert the lists into numpy arrays
        # Loop over a view to allow modification of the original dict as we iterate over it
        for key, value in hac_params_as_dict.items():
            if type(value) == list and key not in ["policies", "her_storage",
                                                   "subgoal_centers", "subgoal_ranges",
                                                   "subgoal_spaces_high", "subgoal_spaces_low"]:
                hac_params_as_dict[key] = np.array(value, dtype=np.float32)

        hac_params = HacParams(**hac_params_as_dict)

    # Load the policies
    saved_policy_dicts = torch.load(f"{directory}/policies.ckpt")
    for level in range(hac_params.num_levels):
        policy_state_dict = saved_policy_dicts[f"policy_level_{level}"]
        hac_params.policies[level].load_state_dict(policy_state_dict)
        hac_params.policies[level].eval()

    return hac_params
