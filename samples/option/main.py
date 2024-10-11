""" import gymnasium as gym
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.option.EUoptionMDP import *
from samples.option.USoptionMDP import *

EUmdp = EUoptionMDP(S0=100, K=100, r=0.05, T=1, sigma=0.2, n=4)
USmdp = USoptionMDP(S0=100, K=100, r=0.05, T=1, sigma=0.2, n=4)
EUsolver = GenericSolver(
    EUmdp,
    simulation_depth_limit=100,
    exploration_constant = 1.0,
    discount_factor = 0.5,
    verbose = False)


EUsolver.run_search(20)
print("European option tree:")
EUsolver.display_tree()

USsolver = GenericSolver(
    USmdp,
    simulation_depth_limit=100,
    exploration_constant = 1.0,
    discount_factor = 0.5,
    verbose = False)

USsolver.run_search(20)
print("American option tree:")
USsolver.display_tree() """

from mcts4py.SolverOption import SolverOption
from samples.option.USoptionMDP import *

mdp = USoptionMDP(option_type = "Put", S0=1, K=0.95, r=0.01, T=5, dt = 1/2, sigma=0.15)
USsolver = SolverOption(
    mdp,
    simulation_depth_limit=100,
    exploration_constant = 1.0,
    discount_factor = 0.5,
    verbose = False)


USsolver.run_option()