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
import time
num_runs = 100

params = config.configurations[config.config_name]

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

timing_dict = {
    "LS": [],
    "UCT": [],
    "UCT VC": [],
    "MENTS": [],
    "MENTS VC": [],
}
mdp = USoptionMDP(option_type=params["option_type"], 
                  S0=params["S0"], 
                  K=params["K"], 
                  r=params["r"], 
                  T=params["T"], 
                  dt=params["dt"], 
                  sigma=params["sigma"], 
                  q = params["q"],
                  price_change="gbm")
MCTS_solver = OptionSolver(
    mdp,
    simulation_depth_limit=params["simulation_depth_limit"],
    exploration_constant=params["exploration_constant"],
    vc = False,
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

MENTS_solver = OptionSolverMENTS(
    mdp,
    exploration_constant=params["exploration_constant"],
    discount_factor =params["discount_factor"],
    temperature = params["temperature"],
    epsilon = params["epsilon"],
    verbose=False
)

ments_solver_2 = StatefulSolverMENTS(
    mdp,
    simulation_depth_limit = 1000,
    discount_factor = 0.6,
    exploration_constant=1.0
    
) 
# Run simulations for Standard MCTS
for i in range(num_runs):
    print(f"Running simulation {i + 1}/{num_runs}")
    #baseline_reward = US_solver.run_baseline()
    #results_dict["Baseline"].append(baseline_reward)
    # ==== UCT ====
    start = time.perf_counter()
    UCT_reward = MCTS_solver.run_option()
    end = time.perf_counter()
    timing_dict["UCT"].append((end - start) * 1000)
    results_dict["UCT"].append(UCT_reward)

    # ==== UCT VC ====
    start = time.perf_counter()
    hindsight_reward = MCTS_solver.run_option_hindsight()
    end = time.perf_counter()
    timing_dict["UCT VC"].append((end - start) * 1000)
    results_dict["UCT VC"].append(hindsight_reward)

    # ==== MENTS ====
    start = time.perf_counter()
    reward_ments = MENTS_solver.run_option()
    end = time.perf_counter()
    timing_dict["MENTS"].append((end - start) * 1000)
    results_dict["MENTS"].append(reward_ments)

    # ==== MENTS VC ====
    start = time.perf_counter()
    reward_ments_vc = MENTS_solver.run_option_hindsight()
    end = time.perf_counter()
    timing_dict["MENTS VC"].append((end - start) * 1000)
    results_dict["MENTS VC"].append(reward_ments_vc)
    #reward_m2 = ments_solver_2.run_option()
    #results_dict["MENTS_2"].append(reward_m2)
    #reward_ments_2_vc = ments_solver_2.run_option_hindsight()
    #print("reward for ments vc",reward_ments_2_vc)
    #results_dict["MENTS_2 VC"].append(reward_ments_2_vc)

    # ==== LS ====
    MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)
    MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)
    MC.stock_price_simulation()

    start = time.perf_counter()
    ls_price_put = MC.american_option_longstaff_schwartz(poly_degree=2, option_type=params["option_type"])
    end = time.perf_counter()
    timing_dict["LS"].append((end - start) * 1000)
    results_dict["LS"].append(ls_price_put)

print("\n=== Empirical Runtime Results (ms) ===")
for algo, times in timing_dict.items():
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"{algo:10s} : {mean_time:.2f} ± {std_time:.2f} ms")
# Convert updated results to DataFrame for plotting
df = pd.DataFrame(results_dict)
#df.to_csv(f'samples/option/output/results_config_{config_name}.csv', index=False)

try:
    df.to_csv(f'samples/option/output/v_3/results_config_{config.config_name}_test.csv', index=False)
    print("File saved successfully: success")
except Exception as e:
    print(f"Failed to save file: {e}")