from samples.atari.GameSolverMENTS import *
from mcts4py.StatefulSolver import *
import matplotlib.pyplot as plt
from samples.atari.GameSolver import *
from samples.atari.GameMDP import *
from samples.atari.config import game_name

results_dict = {
    "Base": [],
    "UCT": [],
    "MENTS":[],
}
num_runs = 15

mdp = GameMDP() 

solverMCTS = GameSolver(
    mdp,
    exploration_constant = 1.0,
    simulation_depth_limit = 1000,
    discount_factor = 0.8,
    env_name = f"ALE/{game_name}",
    verbose = False)

solverMENTS = GameSolverMENTS(
    mdp,
    exploration_constant = 1,
    simulation_depth_limit = 1000,
    discount_factor = 0.9,
    temperature = 1,
    epsilon = 0.1,
    env_name = f"ALE/{game_name}",
    verbose = False)

for _ in range(num_runs):

    reward_base= solverMCTS.run_random_game() 
    results_dict["Base"].append(reward_base)
    reward_mcts= solverMCTS.run_game()
    results_dict["UCT"].append(reward_mcts)
    reward_ments = solverMENTS.run_game() 
    results_dict["MENTS"].append(reward_ments)

df = pd.DataFrame(results_dict)

try:
    df.to_csv(f'samples/option/output/results_config_{game_name}.csv', index=False)
    print("File saved successfully: success")
except Exception as e:
    print(f"Failed to save file: {e}")



