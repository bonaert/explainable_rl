# Explaining Deep Reinforcement Learning policies through Hierarchical Reinforcement Learning

In **reinforcement learning**, an agent is trying to solve a complex task (for example, move a robot or play 
a video game) and has to pick actions which will lead to the maximum amount of reward. Most state-of-the-art
methods use **deep neural networks** to solve very complex tasks, such as folding proteins, designing chips, solving
Go or Starcraft games or doing robotics. Unfortunately, while deep learning often leads to the best performance and can solve 
problems which were impossible bfore, the agents become 
**black box systems** whose internal behavior is mysterious and very hard to understand. It is unclear what the
neural network has learned exactly, how it picks the actions and if it can be relied on. Without this ability, it
is difficult to trust the systems, which makes difficult to use in real-world problems where the stakes matter and mistakes
are costly (not to mention mission-critical systems). 

In this work, we aim to develop **clear, intuitive explanations** for the agent's behavior, which are useful both to
experts and to laymen with little background in the field. To do so, **we construct agents which explain their own behavior**.
To construct the explanations, we leverage one key insight: in human life low-level actions are often part of high-level plans; 
therefore **if we know the agent's high-level goals then the low-level actions become easy to understand: they are simply means to reach the next goal**.

To create these explanations, we train a **hierarchical agent** which has 2 components: one component that create a series of subgoals 
and another which tries to reach those subgoals. The series of subgoals act as the explanation, since they allow us to see
what the agent is trying to do and visualise its long term plan and future behavior. It's a versatile system because
we can place the agent in any state and simulate the goals it would try to reach, without having to actually make the agent act.

Below we show examples of our explanations: one the Mountain Car Environment and the Lunar Lander environment.

### Mountain Car explanation

The red square and black arrow indicate current goal, e.g. the position and speed that the agent is currently trying to reach. The green squares indicate the long term plan of the agent - the next 5 goals the agent will try to reach. The long term plan is updated at every step (its purpose is only for explanation) but the current goal is only updated periodically (so that the agent can reach it). 

![Explanation for the Mountain Car environment](https://github.com/bonaert/explainable_rl/blob/master/media/solved_hac_mountain_car.gif?raw=true)

*Note: the GIF has been slowed down to make it easier to see the goals*

### Lunar Lander explanation

The yellow square and the arrows indicate current goal, e.g. the position, angle and speed that the agent is currently trying to reach. The green squares indicate the long term plan of the agent - the next 10 goals the agent will try to reach. The long term plan is updated at every step (its purpose is only for explanation) but the current goal is only updated periodically (so that the agent can reach it). 

![Explanation for the Lunar Lander environment](https://github.com/bonaert/explainable_rl/blob/master/media/solved_hac_lunar_lander.gif?raw=true)



This work was produced as part of my 2020 Master thesis in Artificial Intelligence / Machine learning, done
at the AI lab of the Vrije Universiteit Brussel (VUB). This repository hosts all the algorithms, parameters, 
saved weights and code I created during the thesis.

## Current status

**High-level summary: we can generate good explanations and train hierarchical agent using the HAC-General With Teacher algorithm. The work is mostly complete, except for finishing touches.**

 - Created 3 different algorithms to train hierarchical agents:
    - **[HAC-General](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/hac_general.py)**: a generalization of hierarchical actor critic with hindsight, which does not require the environment to provide a goal and aims to maximize the reward. Experimental results show that the algorithm by itself cannot train an agent succesfully.
    - **[DAgger-Hierarchical](https://github.com/bonaert/explainable_rl/blob/master/hierarchical_algos/dagger.py)**: a generatlization of DAgger which can train hierarchical agents. The trained hierarchical agent trains fast and achieves great performance, but experiments show that while the explanations tend to be good, they are not always reliable.
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
