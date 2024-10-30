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

# Import necessary modules
from mcts4py.SolverOptionMCTS import SolverOption
from samples.option.USoptionMDPOG import USoptionMDPOG
from samples.option.ls_american.option import Option
from samples.option.ls_american.process import HestonProcess
from samples.option.ls_american.pricing import monte_carlo_simulation_LS
from tabulate import tabulate

# Define the Heston process parameters
heston = HestonProcess(mu=0.06, kappa=0.0005, theta=0.04, eta=0.1, rho=-0.5)

# Define the first put option
put_option_1 = Option(s0=36, v0=0.05, T=1, K=40, call=False)
ls_price_1 = monte_carlo_simulation_LS(option=put_option_1, process=heston, n=25000, m=252)
mdp_1 = USoptionMDPOG(option_type="Put", S0=36, K=40, r=0, T=1, dt=1/10, sigma=0.05)
US_solver_1 = SolverOption(
    mdp_1,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_1 = US_solver_1.run_option()

put_option_3 = Option(s0=36, v0=0.05, T=1, K=40, call=True)
ls_price_3 = monte_carlo_simulation_LS(option=put_option_3, process=heston, n=25000, m=252)
mdp_3 = USoptionMDPOG(option_type="Call", S0=36, K=40, r=0, T=1, dt=1/10, sigma=0.05)
US_solver_3 = SolverOption(
    mdp_3,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_3 = US_solver_3.run_option()

# Define the second put option
put_option_2 = Option(s0=1, v0=0.15, T=1, K=0.95, call=False)
ls_price_2 = monte_carlo_simulation_LS(option=put_option_2, process=heston, n=25000, m=252)
mdp_2 = USoptionMDPOG(option_type="Put", S0=1, K=0.95, r=0, T=1, dt=1/10, sigma=0.15)
US_solver_2 = SolverOption(
    mdp_2,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_2 = US_solver_2.run_option()

put_option_4 = Option(s0=1, v0=0.15, T=1, K=0.95, call=True)
ls_price_4 = monte_carlo_simulation_LS(option=put_option_4, process=heston, n=25000, m=252)
mdp_4 = USoptionMDPOG(option_type="Call", S0=1, K=0.95, r=0, T=1, dt=1/10, sigma=0.15)
US_solver_4 = SolverOption(
    mdp_4,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_4 = US_solver_4.run_option()

# Prepare data for display
data = [
    [36, 40, 'Put',1, 4.01, ls_price_1, mcts_price_1],
    [36, 40, 'Call',1, 0.01, ls_price_3, mcts_price_3],
    [1, 0.95, 'Put',1, 0.04, ls_price_2, mcts_price_2],
    [1, 0.95, 'Call',1, 0.09, ls_price_4, mcts_price_4]
]

# Display results in a formatted table
print(tabulate(data, headers=["S0", "K", "Option Type", "Year","External website", "LS Price", "MCTS Price"], tablefmt="grid"))
