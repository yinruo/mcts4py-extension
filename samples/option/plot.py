import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from samples.option.OptionSolver import OptionSolver
from samples.option.USoptionMDP import USoptionMDP
from samples.option.ls.monte_carlo_class import MonteCarloOptionPricing
from samples.option.OptionSolverMENTS import OptionSolverMENTS
from samples.option.MENTS.SolverMents import StatefulSolverMENTS
import numpy as np

S0 = 40
K = 36
T = 1
r = 0.15
sigma = 0.2
dt = 1/10
div_yield = 0
simulation_depth_limit = 100
exploration_constant = 1.0
num_runs = 1000

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
mdp = USoptionMDP(option_type="Call", S0=S0, K=K, r=r, T=T, dt=dt, sigma=sigma)
US_solver = OptionSolver(
    mdp,
    simulation_depth_limit=100,
    exploration_constant=1.0,
    verbose=False
)

MC = MonteCarloOptionPricing(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        div_yield=div_yield,
        simulation_rounds=1000,
        no_of_slices=91,
        fix_random_seed=np.random.randint(1, 1000000)
    )

ments_solver = OptionSolverMENTS(
    mdp,
    exploration_constant=1.0,
    discount_factor = 0.9,
    temperature = 1,
    epsilon = 0.1,
    verbose=False
)

ments_solver_2 = StatefulSolverMENTS(
    mdp,
    simulation_depth_limit = 1000,
    discount_factor = 0.6,
    exploration_constant=1.0
    
)
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
    ls_price_put = MC.american_option_longstaff_schwartz(poly_degree=2, option_type="put")
    results_dict["LS"].append(ls_price_put)


# Convert updated results to DataFrame for plotting
df = pd.DataFrame(results_dict)

# Plot using seaborn
plt.figure(figsize=(14, 8))
sns.boxplot(data=df)
plt.title("Option Reward Estimates Using Different Methods")
plt.xlabel("Algorithm")
plt.ylabel("Reward")
plt.show()

column_name = "MENTS_2 VC"
q1 = df[column_name].quantile(0.25)
q2 = df[column_name].quantile(0.5) 
q3 = df[column_name].quantile(0.75)

print(f"For {column_name}:")
print("Q1 (25%):", q1)
print("Median (50%):", q2)
print("Q3 (75%):", q3)