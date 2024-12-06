# Import necessary modules
from mcts4py.SolverOptionMCTS import SolverOption
from mcts4py.SolverOptionMENTS import SolverOptionMENTS
from mcts4py.HindsightSolverOptionMCTS import HindsightSolverOption
from mcts4py.TestSolverOptionMCTS import TestSolverOption
from samples.option.USoptionMDP import USoptionMDP
from samples.option.ls.monte_carlo_class import MonteCarloOptionPricing
from mcts4py.ExpectationSolverOptionMCTS import ExpectationSolverOption
from tabulate import tabulate
import numpy as np

# Define data sets with external prices
data_sets = [
    {
        'S0': 1,
        'K': 0.9,
        'T': 1,
        'r': 0,
        'sigma': 0.15,
        'div_yield': 0,
        'external_price_put': 0.02,
        'external_price_call': 0.012
    },
    {
        'S0': 36,
        'K': 40,
        'T': 1,
        'r': 0,
        'sigma': 0.05,
        'div_yield': 0,
        'external_price_put': 4.01,
        'external_price_call': 0.01
    },
    {
        'S0': 10,
        'K': 12,
        'T': 1,
        'r': 0.01,
        'sigma': 0.1,
        'div_yield': 0,
        'external_price_put': 2.00,
        'external_price_call': 0.02
    },
    {
        'S0': 90,
        'K': 100,
        'T': 0.5,
        'r': 0,
        'sigma': 0.15,
        'div_yield': 0,
        'external_price_put': 10.00,
        'external_price_call': 1.87
    }
]

# Prepare results list
results = []

# For each data set
for data_set in data_sets:
    S0 = data_set['S0']
    K = data_set['K']
    T = data_set['T']
    r = data_set['r']
    sigma = data_set['sigma']
    div_yield = data_set['div_yield']
    external_price_put = data_set['external_price_put']
    external_price_call = data_set['external_price_call']

    # Create Monte Carlo object
    MC = MonteCarloOptionPricing(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        div_yield=div_yield,
        simulation_rounds=10000,
        no_of_slices=91,
        fix_random_seed=np.random.randint(1, 10000)
    )
    MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)  # CIR model
    MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)      # Heston model
    MC.stock_price_simulation()

    # Get LS prices for Put and Call options
    ls_price_put = MC.american_option_longstaff_schwartz(poly_degree=2, option_type="put")
    ls_price_call = MC.american_option_longstaff_schwartz(poly_degree=2, option_type="call")

    # For each option type
    for option_type in ['Put', 'Call']:
        # Create MDP object
        mdp = USoptionMDP(option_type=option_type, S0=S0, K=K, r=r, T=T, dt=0.1, sigma=sigma)

        # Run SolverOption
        solver = SolverOption(
            mdp,
            simulation_depth_limit=100,
            exploration_constant=1.0,
            verbose=False
        )
        mcts_price = solver.run_option()

        # Run HindsightSolverOption
        hindsight_solver = HindsightSolverOption(
            mdp,
            simulation_depth_limit=100,
            exploration_constant=1.0,
            verbose=False
        )
        hindsight_price = hindsight_solver.run_option()

        # Run ExpectationSolverOption
        expect_solver = TestSolverOption(
            mdp,
            simulation_depth_limit=100,
            exploration_constant=1.0,
            verbose=False
        )
        expect_price = expect_solver.run_option()

        # Get LS price and external price
        if option_type == 'Put':
            ls_price = ls_price_put
            external_price = external_price_put
        else:
            ls_price = ls_price_call
            external_price = external_price_call

        # Prepare data row
        row = [
            external_price, ls_price, mcts_price,
            hindsight_price, expect_price
        ]

        # Append to results
        results.append(row)

# Display results in a formatted table
print(tabulate(results, headers=[
     "External Price",
    "LS Price", "MCTS", "Hindsight", "MCTS Vanilla"
], tablefmt="grid"))




