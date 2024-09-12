import gymnasium as gym
from copy import deepcopy
from mcts4py.SolverCartpole import *
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.cartpole.cartpoleMDP import *


mdp = cartpoleMDP()

solver = SolverCartpole(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 0.5,
    env_name = "CartPole-v1",
    verbose = False)


solver.run_game(10)