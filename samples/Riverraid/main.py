import gymnasium as gym
from copy import deepcopy
from mcts4py.SolverGame import *
from mcts4py.SolverMENTS import *
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.Riverraid.RiverraidMDP import *
import matplotlib.pyplot as plt

mdp = RiverraidMDP()

solverMCTS = SolverGame(
    mdp,
    exploration_constant = 1.0,
    simulation_depth_limit = 100,
    discount_factor = 0.5,
    env_name = 'ALE/Riverraid-v5',
    verbose = False)


rewards_mcts, average_mcts = solverMCTS.run_game(5)
rewards_random, average_random = solverMCTS.run_random_game(5)

solverMENTS = SolverMENTS(
    mdp,
    simulation_depth_limit = 100,
    exploration_constant = 1,
    discount_factor = 0.9,
    temperature = 1,
    epsilon = 0.1,
    env_name = 'ALE/Riverraid-v5',
    verbose = False)


rewards_ments, average_ments = solverMENTS.run_game(5)  


max_mcts = max(rewards_mcts)
max_random = max(rewards_random)
max_ments = max(rewards_ments)

print(f"MCTS Avg Reward: {average_mcts:.2f}, Max Reward: {max_mcts}")
print(f"Random Avg Reward: {average_random:.2f}, Max Reward: {max_random}")
print(f"MENTS Avg Reward: {average_ments:.2f}, Max Reward: {max_ments}")

plt.plot(rewards_mcts, label="MCTS Rewards", color='blue', marker='o')
plt.plot(rewards_random, label="Random Rewards", color='green', marker='o')
plt.plot(rewards_ments, label="MENTS Rewards", color='red', marker='o')


plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards per Episode")
plt.legend()

plt.grid(True)
plt.show()


