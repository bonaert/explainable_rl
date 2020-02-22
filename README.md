# Explaining Deep Reinforcement Learning policies through Network Distillation

The repository hosts all the algorithms, training parameters, saved weights and code
produced during my 2020 Master thesis in Artificial Intelligence / Machine learning, done
at the AI lab of the Vrije Universiteit Brussel (VUB).

## Current status

For a detailed changelog, please see the [Changelog](CHANGELOG.md). Here's a summary 
of the current status of the thesis:

 - Implemented important RL algorithms, for both discrete and continuous action spaces:
    - REINFORCE / Policy gradient
    - Actor Critic
    - DDPG (Deep Deterministic Policy Gradient)
    - SAC (Soft Actor Critic)
- Can solve important environments, such as:
    - CartPole
    - Pendulum
    - Mountain Car (discrete and continuous)
    - Lunar Lander
    - Bipedal Walker and Bipedal Walker Hardcore
- Implemented the Watershed environment (a custom OpenAI Gym compatible environment). It's
  originally a classical optimization problem but RL will be applied to solve it

![Solved Bipedal Walker Hardcode](https://github.com/bonaert/explainable_rl/blob/master/videos/solved_bipedal_walker_hardcore_sac.gif?raw=true)


## Background
Deep neural networks are advancing the state-of-the-art in the Reinforcement Learning (RL)
framework, a machine learning technique capable of handling difficult sequential decision problems as
well as control problems. This assures that RL will become an indispensable component in the
industry (e.g., manufacturing and socially assistive robotics). 

However, given the complexity of these black-box systems, it is hard to directly understand or explain to laymen why these autonomously
learning systems make certain decisions in the end. In order to further employ reinforcement learning
systems in sensitive environments, for example (semi-)autonomous cars, it is therefore necessary to
enlighten the darkness and have systems that explain their decisions as it forms the basis for
accountability, subsuming the justification of the decisions taken. One possible way is translating deep
neural networks to a more human readable form, better known as Network Distillation.

## Goal of the thesis

This thesis seeks to develop mechanisms to explain learnt Deep Reinforcement Learning
policies through the application of the Network Distillation literature from the Deep Learning
community. Several approaches exist, but we are specifically interested in the application of
decision-tree based approaches such as Soft Decision Trees and Adaptive Neural Trees. However,
additional approaches can be explored at any time.