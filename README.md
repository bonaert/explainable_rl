# Explaining Deep Reinforcement Learning policies through Hierarchical Reinforcement Learning

In **reinforcement learning**, an agent is trying to solve a complex task (for example, move a robot or play 
a video game) and has to pick actions that will lead to the maximum amount of reward. Most state-of-the-art
methods use **deep neural networks** to solve very complex tasks, such as folding proteins, designing chips, solving
Go or Starcraft games or doing robotics. Unfortunately, while deep learning often leads to the best performance and can solve 
problems which were impossible before, the agents become 
**black box systems** whose internal behavior is mysterious and very hard to understand. It is unclear what the
neural network has learned exactly, how it picks the actions, and if it can be relied on. Without this ability, it
is difficult to trust the systems, which makes it difficult to use in real-world problems where the stakes matter and mistakes
are costly (not to mention mission-critical systems). 

In this work, we aim to develop **clear, intuitive explanations** for the agent's behavior, which are useful both to
experts and to laymen with little background in the field. To do so, **we construct agents which explain their own behavior**.
To construct the explanations, we leverage one key insight: in human life low-level actions are often part of high-level plans; 
therefore **if we know the agent's high-level goals then the low-level actions become easy to understand: they are simply means to reach the next goal**.

To create these explanations, we train a **hierarchical agent** which has 2 components: one component that creates a series of subgoals and another which tries to reach those subgoals. The series of subgoals act as the explanation since they allow us to see
what the agent is trying to do and visualize its long-term plan and future behavior. It's a versatile system because
we can place the agent in any state and simulate the goals it would try to reach, without having to actually make the agent act.

Below we show examples of our explanations: one the Mountain Car Environment and the Lunar Lander environment.

### Example 1: Mountain Car goal-based explanation

The red square and black arrow show the current goal, e.g. the position and speed that the agent is trying to reach. The green squares indicate the long-term plan of the agent - the next 5 goals the agent will try to reach. The long-term plan is updated at every step (its purpose is only for explanation) but the current goal is only updated periodically (so that the agent can reach it). 

![Explanation for the Mountain Car environment](https://github.com/bonaert/explainable_rl/blob/master/media/solved_hac_mountain_car.gif?raw=true)

*Note: the GIF has been slowed down to make it easier to see the goals*

### Example 2: Lunar Lander goal-based explanation

The yellow square and the arrows show the current goal, e.g. the position, angle, and speed that the agent is trying to reach. The green squares indicate the long-term plan of the agent - the next 10 goals the agent will try to reach. The long-term plan is updated at every step (its purpose is only for explanation) but the current goal is only updated periodically (so that the agent can reach it). 

![Explanation for the Lunar Lander environment](https://github.com/bonaert/explainable_rl/blob/master/media/solved_hac_lunar_lander.gif?raw=true)

This work was produced as part of my 2020 Master thesis in Artificial Intelligence / Machine learning, done
at the AI lab of the Vrije Universiteit Brussel (VUB). This repository hosts all the algorithms, parameters, 
saved weights, and code I created during the thesis.




## Running the live demonstrations on your computer

If you want to see the hierarchical agent run live on your device instead of a GIF, we have prepared 2 files to do so conveniently:

**Setup**

1. Install the requirements: `pip3 install -r requirements.txt`

2. Install the environments with goal visualisation: consult the  `gym_with_goal_visualisation` folder and its README.

**Mountain Car Demo**:

```bash
cd hierarchical_algos/demos
python3 run_mountain_car_demo.py
```

**Lunar Lander Demo**:

```bash
cd hierarchical_algos/demos
python3 run_lunar_lander_demo.py
```


## Training your own hierarchical agent

**Step 1**: train an expert teacher; we advise using the SAC implementation we provide (see [lunar_lander_continuous.py](https://github.com/bonaert/explainable_rl/blob/master/teacher/scripts/lunar_lander_continuous.py) for an example)

**Step 2**: train the hierarchical agent using the "HAC-General with Teacher" algorithm. We provide a complete working demo in the file [train_demo.py](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/demos/train_demo.py) which you can easily run

```bash
cd hierarchical_algos/demos
python3 train_demo.py
```

The training script is very customizable and you can tweak many parameters. We show below one possible way to train an agent and display most of the parameters you can tweak:

```python
import gym
import numpy as np

# You might need to these imports, see train_demo.py for details
from hac_general import HacParams, train
from teacher.training.sac import get_policy_and_scaler

env = gym.make('LunarLanderContinuous-v2')

# Action space: Low [-1. -1.]	High [1. 1.]
# State space:  Low [-inf] x 8         High [inf] x 8
# State: x, y, vel.x, vel.y, angle, angular_velocity, bool(left left on ground), bool(right leg on ground)
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
```


## Current status

**High-level summary: we can generate good explanations and train hierarchical agent using the HAC-General With Teacher algorithm. The work is mostly complete, except for finishing touches.**

 - Created 3 different algorithms to train hierarchical agents:
    - **[HAC-General](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/hac_general.py)**: a generalization of hierarchical actor-critic with hindsight, which does not require the environment to provide a goal and aims to maximize the reward. Experimental results show that the algorithm by itself cannot train an agent successfully.
    - **[DAgger-Hierarchical](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/dagger.py)**: a generalization of DAgger which can train hierarchical agents. The trained hierarchical agent trains fast and achieves great performance, but experiments show that while the explanations tend to be good, they are not always reliable.
    - **[HAC-General With Teacher](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/hac_general.py)**: this algorithm extends **HAC-General** by using the expert black-box agent to help train the hierarchical agent. This algorithm achieves **both high performance and reliable explanations**, though at the cost of slower training time compared to DAgger-Hierarchical.
 - Create environments which support support visualising goals:
    - [Mountain Car](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/gym/continuous_mountain_car.py)
    - [Lunar Lander](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/gym/lunar_lander.py)
 - Implemented important RL algorithms, for both discrete and continuous action spaces, which can be used to train the black-box expert:
    - [REINFORCE / Policy gradient](https://github.com/bonaert/explainable_rl/blob/master/teacher/training/reinforce.py)
    - [Actor Critic](https://github.com/bonaert/explainable_rl/blob/master/teacher/training/actor_critic.py)
    - [DDPG (Deep Deterministic Policy Gradient)](https://github.com/bonaert/explainable_rl/blob/master/teacher/training/ddpg.py)
    - [SAC (Soft Actor Critic)](https://github.com/bonaert/explainable_rl/blob/master/teacher/training/sac.py)
- Black-box non-hierarchical agents can solve important environments, such as:
    - [CartPole](https://github.com/bonaert/explainable_rl/blob/master/teacher/scripts/cartpole.py)
    - [Pendulum](https://github.com/bonaert/explainable_rl/blob/master/teacher/scripts/pendulum.py)
    - [Mountain Car](https://github.com/bonaert/explainable_rl/blob/master/teacher/scripts/mountaincar.py) (discrete and continuous)
    - [Lunar Lander](https://github.com/bonaert/explainable_rl/blob/master/teacher/scripts/lunar_lander_continuous.py)
    - [Bipedal Walker and Bipedal Walker Hardcore](https://github.com/bonaert/explainable_rl/blob/master/teacher/scripts/bipedal_walker.py)

