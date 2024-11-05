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
from samples.option.ls.monte_carlo_class import MonteCarloOptionPricing
from tabulate import tabulate

#first set of data
S0_1 = 1
K_1=0.9
T_1=1
r_1=0
sigma_1=0.15
div_yield_1=0
#second set of data 
S0_2 = 36
K_2= 40
T_2=1
r_2=0
sigma_2=0.05
div_yield_2=0
# Define the Heston process parameters
MC = MonteCarloOptionPricing(S0=S0_1,
                             K=K_1,
                             T=T_1,
                             r=r_1,
                             sigma=sigma_1,
                             div_yield=div_yield_1,
                             simulation_rounds=int(10000) ,
                             no_of_slices=91 ,
                             # fix_random_seed=True,
                             fix_random_seed=500)
MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)  # use Cox Ingersoll Ross (CIR) model

# stochastic volatility (sigma)
MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)  # heston model

MC.stock_price_simulation()
ls_value = MC.american_option_longstaff_schwartz(poly_degree=2, option_type="put")
ls_value_2 = MC.american_option_longstaff_schwartz(poly_degree=2, option_type="call")

mdp_1 = USoptionMDPOG(option_type="Put", S0=S0_1, K=K_1, r=r_1, T=T_1, dt=1/10, sigma=sigma_1)
US_solver_1 = SolverOption(
    mdp_1,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_1 = US_solver_1.run_option()

mdp_2 = USoptionMDPOG(option_type="Call", S0=S0_1, K=K_1, r=r_1, T=T_1, dt=1/10, sigma=sigma_1)
US_solver_2 = SolverOption(
    mdp_2,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_2 = US_solver_2.run_option()


MC_2 = MonteCarloOptionPricing(S0=S0_2,
                             K=K_2,
                             T=T_2,
                             r=r_2,
                             sigma=sigma_2,
                             div_yield=div_yield_2,
                             simulation_rounds=int(10000) ,
                             no_of_slices=91 ,
                             # fix_random_seed=True,
                             fix_random_seed=500)
MC_2.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)  # use Cox Ingersoll Ross (CIR) model

# stochastic volatility (sigma)
MC_2.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)  # heston model

MC_2.stock_price_simulation()
value_3 = MC_2.american_option_longstaff_schwartz(poly_degree=2, option_type="put")
value_4 = MC_2.american_option_longstaff_schwartz(poly_degree=2, option_type="call")
# Define the first put option
mdp_3 = USoptionMDPOG(option_type="Put", S0=S0_2, K=K_2, r=r_2, T=T_2, dt=1/10, sigma=sigma_2)
US_solver_3 = SolverOption(
    mdp_3,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_3 = US_solver_3.run_option()

mdp_4 = USoptionMDPOG(option_type="Call",  S0=S0_2, K=K_2, r=r_2, T=T_2, dt=1/10, sigma=sigma_2)
US_solver_4 = SolverOption(
    mdp_4,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_4 = US_solver_4.run_option()

# Define the second put option
""" mdp_4 = USoptionMDPOG(option_type="Put", S0=1, K=0.9, r=0, T=1, dt=1/10, sigma=0.15)
US_solver_4 = SolverOption(
    mdp_4,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_4 = US_solver_2.run_option()

mdp_4 = USoptionMDPOG(option_type="Call", S0=1, K=0.9, r=0, T=1, dt=1/10, sigma=0.15)
US_solver_4 = SolverOption(
    mdp_4,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)
mcts_price_4 = US_solver_4.run_option() """

# Prepare data for display
data = [
    [S0_1, K_1, 'Put',r_1, sigma_1, T_1, 0.02, ls_value, mcts_price_1],
    [S0_1, K_1, 'Call',r_1, sigma_1, T_1, 0.012, ls_value_2, mcts_price_2],
    [S0_2, K_2, 'Put',r_2, sigma_2, T_2, 4.01, value_3, mcts_price_3],
    [S0_2, K_2, 'Call',r_2, sigma_2, T_2, 0.01, value_4, mcts_price_4]
]

# Display results in a formatted table
print(tabulate(data, headers=["S0", "K", "Option Type", "Risk-Free Interest Rate","Volatility","Year", "External website", "LS Price", "MCTS Price"], tablefmt="grid"))
