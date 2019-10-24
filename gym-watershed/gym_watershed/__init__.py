from gym.envs.registration import register

register(
    id='watershed-v0',
    entry_point='gym_watershed.envs:WatershedEnv',
)