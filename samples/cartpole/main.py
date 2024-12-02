import gymnasium as gym
from copy import deepcopy
from mcts4py.game.SolverCartpole import *
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.cartpole.cartpoleMDP import *
from mcts4py.game.SolverCartpoleMCTS import *
from mcts4py.game.SolverCartpoleMents import *
import matplotlib.pyplot as plt


mdp = cartpoleMDP()

""" solver = SolverCartpole(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 0.9,
    env_name = "CartPole-v1",
    verbose = True) """


""" solver = SolverCartpoleMCTS(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    env_name = "CartPole-v1",
    verbose = True) """

mean_rewards = []
iterations = range(3, 30)

for i in iterations:
    solver = SolverCartpoleMCTS(
        mdp,
        exploration_constant = 1.0,
        discount_factor = 1,
        env_name = "CartPole-v1",
        iteration_time = i ,
        verbose = False
    )

    mean_reward = solver.run_game(20)
    mean_rewards.append(mean_reward)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(iterations, mean_rewards, marker='o', label='Mean Reward')

# Graph Labels
plt.title("Mean Reward vs Iteration Time", fontsize=16)
plt.xlabel("Iteration Time (i)", fontsize=14)
plt.ylabel("Mean Reward", fontsize=14)
plt.grid(True)
plt.legend()
plt.show()