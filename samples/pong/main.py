from mcts4py.SolverGame import *
from samples.pong.pongMDP import *
from mcts4py.SolverGameMENTS import *
from samples.pong.pongMDP import *
mdp = pongMDP() 
""" solverMCTS = SolverGame(
    mdp,
    exploration_constant = 1.0,
    simulation_depth_limit = 100,
    discount_factor = 0.5,
    env_name = 'ALE/Riverraid-v5',
    verbose = False)

rewards_mcts = solverMCTS.run_game(5) """

mdp = pongMDP() 
solverMents = SolverGameMENTS(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    temperature = 1,
    epsilon = 0.1,
    env_name = 'ALE/Riverraid-v5',
    verbose = False)

rewards_ments = solverMents.run_game(5)


""" games = list(range(1, 6))

# Plotting the rewards for both MCTS and MENTS
plt.figure(figsize=(10, 6))
plt.plot(games, rewards_mcts, marker='o', label='Rewards MCTS', color='b')
plt.plot(games, rewards_ments, marker='s', label='Rewards MENTS', color='g')

# Adding titles and labels
plt.title("Rewards Comparison: MCTS vs MENTS", fontsize=16)
plt.xlabel("Games", fontsize=14)
plt.ylabel("Rewards", fontsize=14)
plt.xticks(games)  # Ensure the x-axis matches the number of games
plt.grid(True)
plt.legend(fontsize=12)

# Display the plot
plt.show() """