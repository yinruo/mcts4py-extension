from samples.cartpole.CartpoleSolver import *
from samples.cartpole.cartpoleMDP import *
from samples.cartpole.CartpoleSolverMENTS import *
import matplotlib.pyplot as plt


mdp = cartpoleMDP()

solver_mcts = CartpoleSolver(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    env_name = "CartPole-v1",
    iteration_time = 100,
    verbose = True)

solver_mcts.run_game(10)

solver_mcts = CartpoleSolverMENTS(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    env_name = "CartPole-v1",
    temperature = 1,
    epsilon = 0.1,
    verbose = False)

solver_mcts.run_game(10)

""" mean_rewards = []
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
plt.show() """