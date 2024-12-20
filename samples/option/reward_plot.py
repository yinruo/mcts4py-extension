""" import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mcts4py.SolverOptionMCTS import SolverOption
from samples.option.USoptionMDPOG import USoptionMDPOG
from mcts4py.HindsightSolverOptionMCTS import HindsightSolverOption
from mcts4py.ExpectationSolverOptionMCTS import ExpectationSolverOption

# Parameters for the option
S0_1 = 1
K_1 = 0.9
T_1 = 1
r_1 = 0
sigma_1 = 0.15
div_yield_1 = 0

# Configuration for simulations
simulation_depth_limit = 100
exploration_constant = 1.0
num_runs = 100

# Data collection for all algorithms
results_dict = {
    "Standard MCTS": [],
    "Hindsight MCTS": [],
    "Expectation MCTS": []
}

# Run simulations for Standard MCTS
for _ in range(num_runs):
    # Initialize MDP and Solver
    mdp = USoptionMDPOG(
        option_type="Put",
        S0=S0_1,
        K=K_1,
        r=r_1,
        T=T_1,
        dt=0.1,
        sigma=sigma_1
    )
    solver = SolverOption(
        mdp,
        simulation_depth_limit=simulation_depth_limit,
        exploration_constant=exploration_constant,
        verbose=False
    )
    # Run solver to get the option price
    option_price = solver.run_option()
    results_dict["Standard MCTS"].append(option_price)

# Run simulations for Hindsight MCTS
for _ in range(num_runs):
    # Initialize MDP and Hindsight Solver
    mdp = USoptionMDPOG(
        option_type="Put",
        S0=S0_1,
        K=K_1,
        r=r_1,
        T=T_1,
        dt=0.1,
        sigma=sigma_1
    )
    hindsight_solver = HindsightSolverOption(
        mdp,
        simulation_depth_limit=simulation_depth_limit,
        exploration_constant=exploration_constant,
        verbose=False
    )
    # Run hindsight solver to get the option price
    hindsight_option_price = hindsight_solver.run_option()
    results_dict["Hindsight MCTS"].append(hindsight_option_price)

# Run simulations for Expectation MCTS
for _ in range(num_runs):
    # Initialize MDP and Expectation Solver
    mdp = USoptionMDPOG(
        option_type="Put",
        S0=S0_1,
        K=K_1,
        r=r_1,
        T=T_1,
        dt=0.1,
        sigma=sigma_1
    )
    expectation_solver = ExpectationSolverOption(
        mdp,
        simulation_depth_limit=simulation_depth_limit,
        exploration_constant=exploration_constant,
        verbose=False
    )
    # Run expectation solver to get the option price
    expectation_option_price = expectation_solver.run_option()
    results_dict["Expectation MCTS"].append(expectation_option_price)

# Convert updated results to DataFrame for plotting
df = pd.DataFrame(results_dict)

# Plot using seaborn
plt.figure(figsize=(12, 7))
sns.boxplot(data=df)
plt.title("Option Price Estimates Using Different MCTS Methods")
plt.xlabel("Algorithm")
plt.ylabel("Option Price Estimate")
plt.show()
 """


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mcts4py.SolverOptionMCTS import SolverOption
from samples.option.USoptionMDP import USoptionMDPOG
from mcts4py.HindsightSolverOptionMCTS import HindsightSolverOption
from mcts4py.ExpectationSolverOptionMCTS import ExpectationSolverOption
from samples.option.ls.monte_carlo_class import MonteCarloOptionPricing
import numpy as np

# Parameters for the option
S0_1 = 36
K_1 = 40
T_1 = 1
r_1 = 0
sigma_1 = 0.05
div_yield_1 = 0

# Configuration for simulations
simulation_depth_limit = 100
exploration_constant = 1.0
num_runs = 100

# Data collection for all algorithms
results_dict = {
    "LS Price": [],
    "Standard MCTS": [],
    "Hindsight MCTS": [],
    "Expectation MCTS": []
}

# Generate LS Price results
for _ in range(num_runs):
    MC = MonteCarloOptionPricing(
        S0=S0_1,
        K=K_1,
        T=T_1,
        r=r_1,
        sigma=sigma_1,
        div_yield=div_yield_1,
        simulation_rounds=1000,
        no_of_slices=91,
        fix_random_seed=np.random.randint(1, 10000)
    )
    MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)  # CIR model
    MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)      # Heston model
    MC.stock_price_simulation()
    ls_price_put = MC.american_option_longstaff_schwartz(poly_degree=2, option_type="put")
    results_dict["LS Price"].append(ls_price_put)

# Run simulations for Standard MCTS
for _ in range(num_runs):
    mdp = USoptionMDPOG(
        option_type="Put",
        S0=S0_1,
        K=K_1,
        r=r_1,
        T=T_1,
        dt=0.1,
        sigma=sigma_1
    )
    solver = SolverOption(
        mdp,
        simulation_depth_limit=simulation_depth_limit,
        exploration_constant=exploration_constant,
        verbose=False
    )
    option_price = solver.run_option()
    results_dict["Standard MCTS"].append(option_price)

# Run simulations for Hindsight MCTS
for _ in range(num_runs):
    mdp = USoptionMDPOG(
        option_type="Put",
        S0=S0_1,
        K=K_1,
        r=r_1,
        T=T_1,
        dt=0.1,
        sigma=sigma_1
    )
    hindsight_solver = HindsightSolverOption(
        mdp,
        simulation_depth_limit=simulation_depth_limit,
        exploration_constant=exploration_constant,
        verbose=False
    )
    hindsight_option_price = hindsight_solver.run_option()
    results_dict["Hindsight MCTS"].append(hindsight_option_price)

# Run simulations for Expectation MCTS
for _ in range(num_runs):
    mdp = USoptionMDPOG(
        option_type="Put",
        S0=S0_1,
        K=K_1,
        r=r_1,
        T=T_1,
        dt=0.1,
        sigma=sigma_1
    )
    expectation_solver = ExpectationSolverOption(
        mdp,
        simulation_depth_limit=simulation_depth_limit,
        exploration_constant=exploration_constant,
        verbose=False
    )
    expectation_option_price = expectation_solver.run_option()
    results_dict["Expectation MCTS"].append(expectation_option_price)

# Convert updated results to DataFrame for plotting
df = pd.DataFrame(results_dict)

# Plot using seaborn
plt.figure(figsize=(14, 8))
sns.boxplot(data=df)
plt.title("Option Price Estimates Using Different Methods")
plt.xlabel("Algorithm")
plt.ylabel("Option Price Estimate")
plt.show()