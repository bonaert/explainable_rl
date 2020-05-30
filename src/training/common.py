from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Union, Optional, Tuple

import gym
import numpy as np
import torch
import sklearn.preprocessing
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
import joblib

from datetime import datetime

from src.networks.simple import SimplePolicyDiscrete, SimplePolicyContinuous, SimpleCritic


def tensor_clamp(x: torch.tensor, minimum: np.ndarray, maximum: np.ndarray) -> torch.Tensor:
    """ Clamps the value of x between maximum and minimum, where are arrays / tensors instead of a single scalar"""
    return torch.max(torch.min(x, torch.as_tensor(maximum)), torch.as_tensor(minimum))


def get_env_name(env: gym.Env) -> str:
    """ Returns the name of the OpenAI Gym environment """
    return env.unwrapped.spec.id


@dataclass
class RunParams:
    """
    Stores the parameters of a training procedure. These are general parameters, which are shared by
    most algorithms. You can specify the training, logging, solvability, model saving frequency, stopping
    conditions, entropy, among others.
    """
    render_frequency: int = 1  # Interval between renders (in number of episodes)
    logging_frequency: int = 1  # Interval between logs (in number of episodes)

    gamma: float = 0.99  # Discount factor
    train_with_batches: bool = False
    batch_size: int = 24

    continuous_actions: bool = True
    should_scale_states: bool = True

    entropy_coeff: float = 1
    entropy_decay: float = 1

    use_tensorboard: bool = False
    tensorboard_log_dir: str = None

    env_can_be_solved: bool = True

    save_model_frequency: int = 0  # How frequently (in episodes) should we save the model (0 = never save)

    stop_at_threshold: bool = True
    maximum_episodes: int = 10000000000000000

    def should_render(self, episode_number: int) -> bool:
        return self.render_frequency > 0 and episode_number % self.render_frequency == 0

    def should_log(self, episode_number: int) -> bool:
        return self.logging_frequency > 0 and episode_number % self.logging_frequency == 0

    def should_save_model(self, episode_number: int):
        return self.save_model_frequency > 0 and episode_number % self.save_model_frequency == 0

    def get_tensorboard_writer(self, env: gym.Env) -> SummaryWriter:
        """ Returns a Tensorboard writer, which allows logging many types of information (scalars, images, ...)"""
        if self.tensorboard_log_dir is not None:
            return SummaryWriter(f"runs/{get_env_name(env)}/{self.tensorboard_log_dir}")
        else:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            return SummaryWriter(f"runs/{get_env_name(env)}/{current_time}")


@dataclass
class TrainingInfo:
    """ Stores information collected during the training process, such as the states, actions, rewards,
     entropies, state values, log probabilities, discount factor, etc..."""
    log_probs: List[float] = field(default_factory=list)
    states: List[Union[np.ndarray, torch.Tensor]] = field(default_factory=list)
    actions: List[Union[np.ndarray, torch.Tensor]] = field(default_factory=list)
    rewards: Union[List[float], torch.Tensor] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)
    state_values: List[torch.Tensor] = field(default_factory=list)
    discounted_rewards: Union[List[float], torch.Tensor] = field(default_factory=list)
    all_discounted_rewards: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    running_reward: float = None
    episode_reward: float = 0
    episode_number: int = 0
    GAMMA: float = 0.99  # The discount factor

    def reset(self):
        """ Clears the state, keeping track of the discounted rewards on a separate tensor"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.discounted_rewards = []
        self.log_probs.clear()
        self.state_values.clear()
        self.episode_reward = 0

    def record_step(self,
                    state: Union[np.ndarray, torch.Tensor],
                    action: Union[np.ndarray, torch.Tensor],
                    reward: float,
                    state_value: Optional[torch.Tensor] = None):
        """ Records the information collected during a step of the trianing process """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.episode_reward += reward

    def update_running_reward(self):
        """ Updates the running reward after the end of an episode """
        if self.running_reward is None:  # First episode ever
            self.running_reward = self.episode_reward
        else:
            self.running_reward = 0.05 * self.episode_reward + 0.95 * self.running_reward

    def compute_discounted_rewards(self):
        """ Compute the discounted rewards of the last episode, normalizing using all the discounted
        reward collected in all past episodes """
        self.episode_number += 1

        # Compute discounted rewards at each step
        self.discounted_rewards = []
        discounted_reward = 0
        for reward in self.rewards[::-1]:
            discounted_reward = reward + self.GAMMA * discounted_reward
            self.discounted_rewards.insert(0, discounted_reward)

        # Normalize the discounted rewards
        self.discounted_rewards = torch.tensor(self.discounted_rewards)
        self.all_discounted_rewards = torch.cat([self.all_discounted_rewards, self.discounted_rewards])
        self.discounted_rewards = (self.discounted_rewards - self.all_discounted_rewards.mean()) / \
                                  (self.all_discounted_rewards.std() + 1e-9)

    def get_batches(self, batch_size: int):
        """ Given the experiences collected during the last episodes, generates batches of
        (states, actions, discounted rewards) """
        permutation = torch.randperm(self.discounted_rewards.shape[0])
        for i in range(0, self.discounted_rewards.shape[0], batch_size):
            indices = permutation[i: i + batch_size]
            states = torch.tensor(self.states)[indices]

            if type(self.actions[0]) == int:
                actions = torch.tensor(self.actions)[indices]
            else:
                actions = torch.cat(self.actions)[indices]
            discounted_rewards = self.discounted_rewards[indices]
            yield states, actions, discounted_rewards


def prepare_state(state: np.ndarray) -> torch.Tensor:
    """ Transform a state into a Torch tensor which can be given to the policies and actors """
    return torch.from_numpy(state).float().unsqueeze(0)


def select_action_discrete(state, policy: SimplePolicyDiscrete, training_info: TrainingInfo):
    """
    Given a policy which outputs probabilities, outputs a discrete action sampled according to the probabilities.
    This functions also logs the entropy of the distribution and the log probability of the sampled action.
    """
    # Get distribution
    state = prepare_state(state)
    probs = policy.forward(state)

    # Sample action and remember its log probability
    m = Categorical(probs)
    action = m.sample()

    training_info.log_probs.append(m.log_prob(action))
    training_info.entropies.append(m.entropy())

    return action.item()


def select_action_continuous(state, policy: SimplePolicyContinuous, training_info: TrainingInfo, env: gym.Env):
    """
    Given a policy which outputs a mean and a standard deviation, constructs a Normal distributions and then
    returns an action sampled from that distribution. This functions also logs the entropy of the distribution
    and the log probability of the sampled action.
    """
    # Get distribution
    state = prepare_state(state)
    mu, sigma = policy.forward(state)

    # Sample action and remember its log probability
    n = Normal(mu, sigma)
    action = n.sample()
    action = tensor_clamp(action, env.action_space.low, env.action_space.high)

    # This is not very clean. TODO: clean this up
    training_info.log_probs.append(n.log_prob(action).sum())
    training_info.entropies.append(n.entropy())

    return action


def get_state_value(state, critic: SimpleCritic):
    """ Transforms the state so that it can be consumed by the critic. The critic
    then evaluate's the states value, which is returned.
    """
    return critic.forward(prepare_state(state))


def get_path(env: gym.Env, filename: str, scripts_dir_path: Path):
    """ Given the scripts directory path, finds the path of the data directory then returns the path
            'data/{environment name}/{filename}'
    """
    path = scripts_dir_path.parent.parent / 'data' / get_env_name(env) / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    full_path = path.resolve().as_posix()
    return full_path


def save_numpy(values: torch.Tensor, env: gym.Env, filename: str):
    """ Saves a Numpy array at the path 'data/{environment name}/{filename}' """
    np.save(get_path(env, filename, Path(__file__).parent), values)


def save_tensor(values: torch.Tensor, env: gym.Env, filename: str):
    """ Saves a Pytorch Tensor at the path 'data/{environment name}/{filename}' """
    torch.save(values, get_path(env, filename, Path(__file__).parent))


def save_model(model: torch.nn.Module, env: gym.Env, filename: str):
    """ Saves the Pytorch model at the path 'data/{environment name}/{filename}' """
    torch.save(model.state_dict(), get_path(env, filename, Path(__file__).parent))


def save_scaler(scaler, env: gym.Env, filename: str):
    """ Saves the Scikit-learn scaler at the path 'data/{environment name}/{filename}' """
    joblib.dump(scaler, get_path(env, filename, scripts_dir_path=Path(__file__).parent))


def load_numpy(env: gym.Env, filename: str):
    """ Loads the Numpy array at the path 'data/{environment name}/{filename}' """
    return np.load(get_path(env, filename, Path(__file__).parent))


def load_tensor(env: gym.Env, filename: str):
    """ Loads the Pytorch Tensor at the path 'data/{environment name}/{filename}' """
    full_path = get_path(env, filename, scripts_dir_path=Path(__file__).parent)
    return torch.load(full_path).clone().detach()


def load_model(model_to_fill: torch.nn.Module, env: gym.Env, filename: str):
    """ Loads into the model to fill that weights that can be found at the path 'data/{environment name}/{filename}' """
    full_path = get_path(env, filename, scripts_dir_path=Path(__file__).parent)
    model_to_fill.load_state_dict(torch.load(full_path))
    model_to_fill.eval()
    return model_to_fill


def load_scaler(env: gym.Env, filename: str):
    """ Returns the Scikit-learn scaler that can be found at the path 'data/{environment name}/{filename}' """
    full_path = get_path(env, filename, Path(__file__).parent)
    return joblib.load(full_path)


def setup_observation_scaler(env: gym.Env) -> sklearn.preprocessing.StandardScaler:
    """ Samples many observation from the environment and then creates a Scikit-learn Standard scaler
    that can normalize observations, which is then returned. """
    observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples.squeeze())
    return scaler


def scale_state(scaler: sklearn.preprocessing.StandardScaler, state: np.ndarray) -> np.ndarray:
    """ Scales the state given the scaler """
    return scaler.transform(state.reshape(1, -1))[0]


def run_policy_repeatedly(env: gym.Env, policy: torch.nn.Module, scaler: sklearn.preprocessing.StandardScaler = None,
                          render=True):
    """ Using the policy, actions are taken in the environment until the end of an episode. Each time an episode
    ends, the environment is reset and the process start again. Information is logged on the console and the
    environment can be rendered optionally (useful for visualisation of the policy and debugging)"""
    episode_number = 0
    episode_rewards = []
    while True:
        episode_reward, episode_length = run_general_policy(policy, env, scaler, render)
        episode_rewards.append(episode_reward)

        print(f"Episode {episode_number}\t"
              f"Reward: {episode_reward:.3f}\t"
              f"Number of steps: {episode_length}\t"
              f"Avg reward: {np.mean(episode_rewards):.3f} +- {np.std(episode_rewards):.3f}")
        episode_number += 1


def run_general_policy(
        policy: torch.nn.Module,
        env: gym.Env,
        continuous_actions: bool = True,
        render: bool = True,
        scaler=None) -> Tuple[float, int]:
    """ Runs the policy on the environment until the end of the episode. The environment can
    have either a continuous or a discrete action space. If needed, the states / observations
    can be scaled and the environment can be rendered at each step.
    The function returns the total reward collected during the episode and the episode length.
    """
    training_info = TrainingInfo()

    episode_length, episode_reward = 0, 0
    state = env.reset()

    for t in range(env.spec.max_episode_steps):
        state = scale_state(scaler, state) if scaler is not None else state
        if continuous_actions:
            action = select_action_continuous(state, policy, training_info, env)
        else:
            action = select_action_discrete(state, policy, training_info)

        new_state, reward, done, _ = env.step(action)

        if render:
            env.render()

        state = new_state
        episode_reward += reward
        episode_length += 1

        if done:
            break

    return episode_reward, episode_length


def policy_run(env: gym.Env, policy: torch.nn.Module, scaler: sklearn.preprocessing.StandardScaler = None,
               render=True, algo_is_sac=False, run_once=False, get_watershed_info=False):
    """ Runs the policy on the environment, doing an infinite amount of episodes. The action space must be continuous.
        If needed, the states / observations can be scaled and the environment can be rendered at each step. Some
        logging is done on the console, to understand the behavior and results of the policy.
        """
    solutions = {}
    with torch.no_grad():
        episode_number = 0
        episode_rewards = []
        last_rewards = []
        while True:
            if get_watershed_info:
                state = env.reset(scenario_number=episode_number)
            else:
                state = env.reset()
            done, episode_reward, episode_length = False, 0, 0

            # Update policy bounds
            # if algo_is_sac:
            #     policy.action_high = torch.tensor(env.action_space.high)
            #     policy.action_low = torch.tensor(env.action_space.low)

            while not done:
                if scaler:
                    state = scale_state(scaler, state)

                if algo_is_sac:
                    action = policy.get_actions(torch.tensor(state).float(), deterministic=True)
                else:
                    action = policy.get_actions(torch.tensor(state).float())
                action = np.clip(action, env.action_space.low, env.action_space.high)

                state, reward, done, _ = env.step(action)
                if render:
                    env.render()
                episode_reward += reward
                episode_length += 1
            episode_rewards.append(episode_reward)
            last_rewards.append(reward)

            solutions[episode_number] = (reward, list(action))

            print(f"Episode {episode_number}\t"
                  f"Reward: {episode_reward:.2f} \t"
                  f"Number of steps: {episode_length} \t"
                  f"Avg reward: {np.mean(episode_rewards):.2f} +- {np.std(episode_rewards):.1f} \t"
                  f"Last reward: {reward:.2f} \t"
                  f"Avg Last reward: {np.mean(last_rewards):.2f} +- {np.std(last_rewards):.1f}")
            episode_number += 1

            if run_once:
                return
            elif get_watershed_info and episode_number == 150:
                return solutions






def log_on_tensorboard(env, episode_number, reward, run_params, num_episode_steps, training_info, writer):
    """ If Tensorboard logging should be done at the current episode, statistics about the current episode and the
    general state of the training procedure are logged."""
    if run_params.use_tensorboard:
        if run_params.env_can_be_solved:
            writer.add_scalar("Data/Solved", num_episode_steps < env.spec.max_episode_steps - 1, episode_number)

        writer.add_scalar("Data/Average reward", float(training_info.episode_reward) / num_episode_steps,
                          episode_number)
        writer.add_scalar("Data/Episode reward", float(training_info.episode_reward), episode_number)
        writer.add_scalar("Data/Running reward", float(training_info.running_reward), episode_number)
        writer.add_scalar("Data/Last reward", float(reward), episode_number)
        writer.add_scalar("Data/Number of steps per episode", num_episode_steps, episode_number)
        writer.flush()


def log_on_console(env: gym.Env, episode_number: int, reward: float,
                   run_params: RunParams, num_episode_steps: int, training_info: TrainingInfo):
    """ If logging should be done at the current episode, statistics about the current episode and the
    general state of the training procedure are logged."""
    if run_params.should_log(episode_number):
        solvedMessage = f"Solved: {num_episode_steps < env.spec.max_episode_steps - 1}\t" if run_params.env_can_be_solved else ""
        print(f"Episode {episode_number}\t"
              f"{solvedMessage}"
              f"Avg. reward: {float(training_info.episode_reward) / num_episode_steps:.2f}\t"
              f"Episode reward: {float(training_info.episode_reward):.2f}\t"
              f"Running Reward: {float(training_info.running_reward):.2f}\t"
              f"Last Reward: {float(reward):.2f}\t"
              f"# Steps in episode: {num_episode_steps}")


def close_tensorboard(run_params: RunParams, writer: SummaryWriter):
    """ If Tensorboard was active, everything is flushed to disk and then the writer is closed.
    This ensures all the data will be written and that there are no resource leaks. """
    if run_params.use_tensorboard:
        writer.flush()
        writer.close()


def polyak_average(source_network: torch.nn.Module, target_network: torch.nn.Module, polyak_coeff: float):
    """ Given two networks with the same architecture, updates the parameters in the target network
    using the parameters of the source network, according to the formula:
        param_target := param_target * polyak_coeff + (1 - polyak_coeff) * param_source
    """
    with torch.no_grad():
        for param, param_target in zip(source_network.parameters(), target_network.parameters()):
            # We use the inplace operators to avoid creating a new tensor needlessly
            param_target.data.mul_(polyak_coeff)
            param_target.data.add_((1 - polyak_coeff) * param.data)
