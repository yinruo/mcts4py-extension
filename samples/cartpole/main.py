import gymnasium as gym
from copy import deepcopy
from mcts4py.SolverCartpole import *
from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from samples.cartpole.cartpoleMDP import *
from mcts4py.SolverMENTS_copy import *
from mcts4py.SolverCartpoleTest import *
from mcts4py.SolverCartpole2 import *
from mcts4py.SolverCartpoleMents import *


mdp = cartpoleMDP()

""" solver = SolverMENTS_copy(
    mdp,
    simulation_depth_limit = 100,
    exploration_constant = 1.0,
    discount_factor = 0.9,
    temperature = 0.5,
    epsilon = 0.1,
    env_name = "CartPole-v1",
    verbose = False) """

""" solver = SolverCartpole(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 0.9,
    env_name = "CartPole-v1",
    verbose = True) """

""" solver = SolverCartpoleTest(
    mdp,
    exploration_constant = 1.0,
    simulation_depth_limit = 100,
    discount_factor = 1,
    env_name = "CartPole-v1",
    verbose = True) """

""" solver = SolverCartpole2(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    env_name = "CartPole-v1",
    verbose = True) """

solver = SolverCartpoleMents(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    temperature = 0.5,
    epsilon = 0.2,
    env_name = "CartPole-v1",
    verbose = False)
solver.run_game(7)
