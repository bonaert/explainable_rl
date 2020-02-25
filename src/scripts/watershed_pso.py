import gym
import gym_watershed
import pyswarms as ps
from pyswarms.backend.topology import VonNeumann
import numpy as np
import pyswarm

if __name__ == "__main__":
    # env = gym.make('watershed-v0', limited_scenarios=True, increment_actions=False, bizarre_states=False)
    env = gym.make('watershed-v0', limited_scenarios=False, increment_actions=False, bizarre_states=False)


    def evaluate_action(action):
        _, reward, _, _ = env.step(np.array(action))
        return -reward


    def evaluate_all_actions(actions):
        return [evaluate_action(action) for action in actions]


    num_episodes = 1000
    fitness_list, optimal_values = [], []
    for i in range(num_episodes):
        env.reset()

        # Using the more advanced PSO doesn't change anything
        # optimizer = ps.single.GeneralOptimizerPSO(
        #     n_particles=100,
        #     dimensions=4,
        #     options={
        #         'c1': 2.05,
        #         'c2': 2.05,
        #         'w': 0.7298,
        #         'r': 2,  # range of the Von Neumann topology
        #         'p': 1,  # use L1 or L2 distance
        #     },
        #     bounds=(env.flows_lower_bounds, env.flows_upper_bounds),
        #     topology=VonNeumann(static=False))
        # optimum_fitness, optimal_vars = optimizer.optimize(evaluate_all_actions, iters=1000)

        optimal_vars, optimum_fitness = pyswarm.pso(evaluate_action, env.flows_lower_bounds, env.flows_upper_bounds)

        fitness_list.append(-optimum_fitness)
        optimal_values.append(optimal_vars)
        print(f"Fitness: {-optimum_fitness:.3f}\t"
              f"Optimal values: {[round(x, 3) for x in optimal_vars]}\t"
              f"Average Fitness: {np.mean(fitness_list):.2f} +- {np.std(fitness_list):.2f}")

    print(f"Average Fitness: {np.mean(fitness_list):.2f} +- {np.std(fitness_list):.2f}")

    env.close()
