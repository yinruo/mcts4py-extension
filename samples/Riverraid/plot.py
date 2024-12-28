
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from samples.Riverraid.RiverraidMDP import *
from samples.Riverraid.GameSolver import *
from samples.Riverraid.GameSolverMENTS import *
from samples.Riverraid.GameSolverSR import *

num_runs = 5

results_dict = {
    "Vanilla MCTS": [],
    "1/2-greedy + UCT MCTS": [],
    "MENTS": [],
}
mdp = RiverraidMDP() 

solverMCTS = GameSolver(
        mdp,
        exploration_constant = 1.0,
        simulation_depth_limit = 100,
        discount_factor = 0.5,
        env_name = 'ALE/Riverraid-v5',
        verbose = False)
results_dict["Vanilla MCTS"] = solverMCTS.run_game(num_runs)

solverMents = GameSolverMENTS(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    temperature = 1,
    epsilon = 0.1,
    env_name = 'ALE/Riverraid-v5',
    verbose = False)

results_dict["MENTS"] = solverMents.run_game(num_runs)

solverSR = GameSolverSR(
    mdp,
    exploration_constant = 1.0,
    simulation_depth_limit = 100,
    discount_factor = 0.5,
    env_name = 'ALE/Riverraid-v5',
    verbose = False
)
results_dict["1/2-greedy + UCT MCTS"] = solverSR.run_game(num_runs) 


df = pd.DataFrame(results_dict)

# Plot using seaborn
plt.figure(figsize=(14, 8))
sns.boxplot(data=df)
plt.title("Riverraid Reward Estimates Using Different Methods")
plt.xlabel("Algorithm")
plt.ylabel("Reward Estimate")
plt.show()