import gymnasium as gym
from copy import deepcopy
from mcts4py.SolverCartpole import *
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.cartpole.cartpoleMDP import *
from mcts4py.SolverCartpoleMCTS import *
from mcts4py.SolverCartpoleMents import *


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

solver = SolverCartpoleMents(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    temperature = 1,
    epsilon = 0.1,
    env_name = "CartPole-v1",
    verbose = False)

solver.run_game(7)
