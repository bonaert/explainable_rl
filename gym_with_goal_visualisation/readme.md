# Envs with goal visualisation

To visualise the goals, I created variants of the environments which
take goals during the render() call and show them too.

## Installation

Currently, to add the visualisation, the system is a bit clunky: you have to copy these files into the 
gym package, replacing the ones currently present. Here are the steps to do so:

 1. Find where the `gym` package is on your file system and go to the `envs/classic_control` folder.

    On my Linux, for Python3.8, you can find those files at `~/.local/lib/python3.8/site-packages/gym/envs/classic_control`
or `/usr/local/lib/python3.8/site-packages/gym/envs/classic_control`.

 2. Copy each file of the `gym_with_goal_visualisation` folder into the `classic_control` folder.
 
 3. That's it, you're done!

## How to use

### Lunar Lander

**No goals**

```python
env.unwrapped.render()
```

**With current goal (yellow)**

```python
env.unwrapped.render(state=state, goal=goal)
```

**With current goal (yellow) and long term plan (green)**

```python
env.unwrapped.render(state=current_state, goal=goal, plan_subgoals=list_of_subgoals)
```


### Mountain Car

**No goals**

```python
env.unwrapped.render()
```

**With current goal (yellow)**

```python
env.unwrapped.render(goal=goal)
```

**With current goal (yellow) and long term plan (green)**

```python
env.unwrapped.render(goal=goal, plan_subgoals=list_of_subgoals)
```

**With current goal (yellow), long term plan (green) and end goal (state you want to reach)**

```python
env.unwrapped.render(goal=goal, end_goal=end_goal, plan_subgoals=list_of_subgoals)
```
