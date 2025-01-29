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

""" solver_mcts = CartpoleSolverMENTS(
    mdp,
    exploration_constant = 1.0,
    discount_factor = 1,
    env_name = "CartPole-v1",
    temperature = 1,
    epsilon = 0.1,
    verbose = False)

solver_mcts.run_game(10) """

