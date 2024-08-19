import gymnasium as gym
from copy import deepcopy
from mcts4py.GenericGameSolver import *
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.lunarlander.lunarlanderMDP import *


mdp = lunarlanderMDP()

solver = GenericGameSolver(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 0.5,
    verbose = False)


solver.run_game(10)


