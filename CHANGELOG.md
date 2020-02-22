
14-22 February
-----------

- Implemented Actor Critic, DDPG and SAC
- Can now solve many environments:
    - Pendulum
    - Mountain Car Continuous
    - Lunar Lander Continuous
    - Bipedal Walker
    - Bipedal Walker Hardcore
- Observations can now be scaled / normalized before giving them to the agent
- During the first X steps, the actions can be random (if desired)
- Fixed bugs in the Watershed environment
- Many practical features were added:
    - Saving the weights periodically to disk
    - During training, regularly test the agent's performance to see its evolution
    - Log training statistics and information on Tensorboard
    - During training, to better understand the actions the agent is doing, the 
    environment can be rendered every X episodes
    
![Solved Bipedal Walker Hardcode](/videos/android_vid_test.gif)

12 February
-----------

- Implemented the Reinforce algorithm (simple policy gradient)
- Able to solve the CartPole-v1 environment in 400 steps

23 October
----------

- Finished implementing the Watershed environment as a fully compatible OpenAI gym