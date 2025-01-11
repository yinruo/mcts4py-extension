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
num_runs = 500
config_name = "F"
params = config.configurations[config_name]

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
mdp = USoptionMDP(option_type=params["option_type"], 
                  S0=params["S0"], 
                  K=params["K"], 
                  r=params["r"], 
                  T=params["T"], 
                  dt=params["dt"], 
                  sigma=params["sigma"], 
                  q = params["q"])
US_solver = OptionSolver(
    mdp,
    simulation_depth_limit=params["simulation_depth_limit"],
    exploration_constant=params["exploration_constant"],
    verbose=False
)

MC = MonteCarloOptionPricing(
        S0=params["S0"],
        K=params["K"],
        T=params["T"],
        r=params["r"],
        sigma=params["sigma"],
        div_yield=params["q"],
        simulation_rounds=5,
        no_of_slices=91,
        fix_random_seed=np.random.randint(1, 1000000)
    )

ments_solver = OptionSolverMENTS(
    mdp,
    exploration_constant=params["exploration_constant"],
    discount_factor =params["discount_factor"],
    temperature = params["temperature"],
    epsilon = params["epsilon"],
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

    MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1) 
    MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5) 
    MC.stock_price_simulation()
    ls_price_put = MC.american_option_longstaff_schwartz(poly_degree=2, option_type=params["option_type"])
    results_dict["LS"].append(ls_price_put)


# Convert updated results to DataFrame for plotting
df = pd.DataFrame(results_dict)
#df.to_csv(f'samples/option/output/results_config_{config_name}.csv', index=False)

try:
    df.to_csv(f'samples/option/output/results_config_{config_name}.csv', index=False)
    print("File saved successfully: success")
except Exception as e:
    print(f"Failed to save file: {e}")