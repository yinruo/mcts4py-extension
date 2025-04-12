import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from samples.option.Multi.MultiOptionSolver import MultiOptionSolver
from samples.option.Multi.USMultiOptionMDP import USMultiOptionMDP
from samples.option.ls.monte_carlo_class import MonteCarloOptionPricing
from samples.option.Multi.MultiOptionSolverMENTS import MultiOptionSolverMENTS
import numpy as np
import samples.option.config as config
num_runs = 100

params = config.multi_option_configurations[config.multi_option_config_name]

results_dict = {
    "LS":[],
    "UCT": [],
    "UCT VC": [],
    "MENTS":[],
    "MENTS VC":[],
}


mdp = USMultiOptionMDP(
    option_type_list=params["option_type_list"],
    S0_list=params["S0_list"],
    K_list=params["K_list"],
    r=params["r"],
    T=params["T"],
    dt=params["dt"],
    sigma_list=params["sigma_list"],
    q_list=params["q_list"],
    max_exercise_per_step=params["max_exercise_per_step"]
)


MCTS_solver = MultiOptionSolver(
    mdp,
    simulation_depth_limit=params["simulation_depth_limit"],
    exploration_constant=params["exploration_constant"],
    vc = False,
    verbose=False
) 

MENTS_solver = MultiOptionSolverMENTS(
    mdp,
    exploration_constant=params["exploration_constant"],
    discount_factor =params["discount_factor"],
    temperature = params["temperature"],
    epsilon = params["epsilon"],
    verbose=False
)

 
# Run simulations for Standard MCTS
for i in range(num_runs):
    print(f"Running simulation {i + 1}/{num_runs}")
    #baseline_reward = US_solver.run_baseline()
    #results_dict["Baseline"].append(baseline_reward)
    # ==== UCT ====
    UCT_reward = MCTS_solver.run_option()
    results_dict["UCT"].append(UCT_reward)

    # ==== UCT VC ====
    hindsight_reward = MCTS_solver.run_option_hindsight()
    results_dict["UCT VC"].append(hindsight_reward)


    # === LSM===
    ls_total = 0.0
    for j in range(len(params["S0_list"])):
        MC = MonteCarloOptionPricing(
            S0=params["S0_list"][j],
            K=params["K_list"][j],
            T=params["T"],
            r=params["r"],
            sigma=params["sigma_list"][j],
            div_yield=params["q_list"][j],
            simulation_rounds=5,
            no_of_slices=91,
            fix_random_seed=np.random.randint(1, 1000000)
        )
        MC.cox_ingersoll_ross_model(a=0.5, b=0.05, sigma_r=0.1)
        MC.heston(kappa=2, theta=0.3, sigma_v=0.3, rho=0.5)
        MC.stock_price_simulation()

        ls_price = MC.american_option_longstaff_schwartz(
            poly_degree=2,
            option_type=params["option_type_list"][j]
        )
        ls_total += ls_price
    print("ls_total", ls_total)

    results_dict["LS"].append(ls_total)


    # ==== MENTS ====
    reward_ments = MENTS_solver.run_option()
    results_dict["MENTS"].append(reward_ments)

    # ==== MENTS VC ====
    reward_ments_vc = MENTS_solver.run_option_hindsight()
    results_dict["MENTS VC"].append(reward_ments_vc)


print("\n=== Empirical Runtime Results (ms) ===")

# Convert updated results to DataFrame for plotting
df = pd.DataFrame(results_dict)
#df.to_csv(f'samples/option/output/results_config_{config_name}.csv', index=False)

try:
    df.to_csv(f'samples/option/output/v_3/results_multi_{config.multi_option_config_name}_test.csv', index=False)
    print("File saved successfully: success")
except Exception as e:
    print(f"Failed to save file: {e}")