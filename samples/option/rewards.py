import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from samples.option.OptionSolver import OptionSolver
from samples.option.USoptionMDP import USoptionMDP
from samples.option.ls.monte_carlo_class import MonteCarloOptionPricing
from samples.option.OptionSolverMENTS import OptionSolverMENTS
from samples.option.MENTS.SolverMents import StatefulSolverMENTS
import numpy as np
import samples.option.config as config
num_runs = 100


# Data collection for all algorithms
results_dict = {
    #"Baseline": [],
    "LS": [],
    "UCT": [],
    "UCT VC": [],
    "MENTS":[],
    "MENTS VC":[],
    #"MENTS_2":[],
    #"MENTS_2 VC": []
}
mdp = USoptionMDP(option_type=config.option_type, S0=config.S0, K=config.K, r=config.r, T=config.T, dt=config.dt, sigma=config.sigma, q = config.q)
US_solver = OptionSolver(
    mdp,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)

MC = MonteCarloOptionPricing(
        S0=config.S0,
        K=config.K,
        T=config.T,
        r=config.r,
        sigma=config.sigma,
        div_yield=config.q,
        simulation_rounds=5,
        no_of_slices=91,
        fix_random_seed=np.random.randint(1, 1000000)
    )

ments_solver = OptionSolverMENTS(
    mdp,
    exploration_constant=1.0,
    discount_factor = 0.9,
    temperature = 0.7,
    epsilon = 0.2,
    verbose=False
)

""" ments_solver_2 = StatefulSolverMENTS(
    mdp,
    simulation_depth_limit = 1000,
    discount_factor = 0.6,
    exploration_constant=1.0
    
) """
# Run simulations for Standard MCTS
for _ in range(num_runs):

    #baseline_reward = US_solver.run_baseline()
    #results_dict["Baseline"].append(baseline_reward)
    UCT_reward = US_solver.run_option()
    results_dict["UCT"].append(UCT_reward)
    hindsight_reward =  US_solver.run_option_hindsight()
    results_dict["UCT VC"].append(hindsight_reward)
    reward_ments = ments_solver.run_option()
    results_dict["MENTS"].append(reward_ments)
    reward_ments_vc = ments_solver.run_option_hindsight()
    results_dict["MENTS VC"].append(reward_ments_vc)
    #reward_m2 = ments_solver_2.run_option()
    #results_dict["MENTS_2"].append(reward_m2)
    #reward_ments_2_vc = ments_solver_2.run_option_hindsight()
    #print("reward for ments vc",reward_ments_2_vc)
    #results_dict["MENTS_2 VC"].append(reward_ments_2_vc)

    MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)  # CIR model
    MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)      # Heston model
    MC.stock_price_simulation()
    ls_price_put = MC.american_option_longstaff_schwartz(poly_degree=2, option_type=config.option_type)
    results_dict["LS"].append(ls_price_put)


# Convert updated results to DataFrame for plotting
df = pd.DataFrame(results_dict)
df.to_csv('samples/option/output/results_config_b.csv', index=False)